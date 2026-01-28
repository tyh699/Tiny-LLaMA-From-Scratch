import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# 1. 配置类
# 作用：把模型所有的超参数都放在一个地方方便管理
@dataclass
class ModelArgs:
    dim: int = 4096 #模型隐藏层的维度
    n_layers: int = 32 #Transformer 层数
    n_heads: int = 32 #多头注意力的头数
    n_kv_heads: Optional[int] = None #KV Cache的头数（用于GQA，若为None则等于 n_heads）
    vocab_size: int = -1 #词表大小（通常在加载tokenizer后设置）
    multiple_of: int = 256 #FFN隐藏层维度的倍数（用于SwiGLU维度对齐）
    ffn_dim_multiplier: Optional[float] = None #用于微调FFN中间层大小的系数
    norm_eps: float = 1e-5 #RMSNorm 的epsilon 防止分母为0
    max_seq_len: int = 2048 #最大序列长度 上下文窗口大小，模型一次最多能看多少字
    dropout: float = 0.0 #Dropout 概率 防止过拟合的机制，训练时随机丢弃一些神经元


# 2.归一化层（RMSNorm）
# 作用：让数据分布更稳定，防止梯度爆炸或消失
# 区别：比传统的LayerNorm少减了一个均值Mean，计算更快，效果差不多
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        #这是一个可学习的缩放参数，模型会自动调整它来放大或缩小模型
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        # 公式： x*(1/sqrt(mean(x^2) + eps))  $$\bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum x_i^2 + \epsilon}}$$
        # x.pow(2):所有数平方
        # mean(-1):在最后一个维度求均值
        # rsqrt:平方根的倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self,x):
        # 技巧：先转成float32进行计算（保证精度），算完再转回原来的模型（比如bfloat16）
        output = self._norm(x.float()).type_as(x)
        # 最后乘上课学习的缩放参数
        return output * self.weight
    

# 3.旋转位置编码（RoPE）辅助函数
# 作用：告诉模型每个词在句子里的位置
# 原理：通过旋转向量的角度来表示相对位置
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 预计计算旋转角度（复数形式） dim：每个头的维度（head_dim） end：最大序列长度（max_seq_len）
    # 计算频率：1/theta^(2i/dim)
    # 这里的切片：[: (dim // 2)] 是因为复数需要两个实数表示，所以只需要一半的维度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成位置索引序列：[0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)  # 创建一个长度为end的序列
    # 外积计算：生成所有位置对应的频率
    # 结果shape：(seq_len, dim / 2)
    freqs = torch.outer(t, freqs).float()  # 矩阵乘法

    # 将模长设为1，角度设为freqs，生成复数(cos+i*sin)
    # 结果是一个复数张量
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 形状转换：将旋转向量广播到与输入张量的形状一致
    # 调整频率矩阵的形状，让他能和输入x进行广播（自动对齐）
    # 目标：让freqs_cis变成(1,seq_len,head_dim/2)
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1],x.shape[-1])
    # 构造新形状：除了seq_len 和 head_dim 维度，其他维度都设为1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    # 真正执行旋转操作的地方，xq：query向量，xk：key向量，freqs_cis：旋转角度
    # 把实数转成复数形式，比如shape从(...,dim)变成(...,dim/2)，每个元素是一个复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1],-1,2))

    # 调整频率矩阵形状以匹配xq
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 复数乘法 = 旋转操作
    # 这一步把位置信息注入到了query和key中
    # flatten(3)是把复数再展平回实数：(...,dim/2) -> (...,dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 转回原来的数据类型返回
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 4.注意力机制 这是transformer的心脏
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 处理GQA（分组查询注意力）逻辑
        # 如果没设置n_kv_heads，就默认和n_heads一样，这是标准多头注意力
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads

        # 这里的local变量是为了兼容多卡分布式训练，当前代码我们只用单卡，所以等于total
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads

        # 计算每个头的维度 例如 dim=4096,heads=32，那么head_dim=128
        self.head_dim = args.dim // args.n_heads

        # 计算KV需要重复几次，比如Query有32个头，KV只有8个头，那么KV每个头需要重复4次才能匹配
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 定义4个线性层（全连接层）
        # 也就是公式里的Wq,Wk,Wv,Wo
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False) # 注意维度可能比Q小
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False) # 输出层
        
        self.dropout = args.dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x.shape = (batch_size, seq_len, dim)
        bsz, seqlen, _ = x.shape

        # 投影：把输入x变成Q，K，V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 分头：把长向量切成多个头
        # view之后shape：(batch_size, seq_len, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 旋转位置编码（RoPE）：给Q和K加上位置信息，注意：V不需要加位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # GQA处理：如果KV头数少，需要复制扩展
        if self.n_rep > 1:
            # repeat_interleave: 在dim=2（heads维度）复制n_rep次
            xk = torch.repeat_interleave(xk, self.n_rep, dim=2)
            xv = torch.repeat_interleave(xv, self.n_rep, dim=2)
        
        # 转置：为了做矩阵乘法，把Heads移到前面 shape变成：(Batch,Heads,Seq,Head_Dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 计算注意力分数scores
        # 公式：Q @ K.T / sqrt(dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 应用Mask
        # 作用：让模型看不见未来的词，把未来的位置分数设为负无穷大
        if mask is not None:
            scores = scores + mask
        
        # Softmax归一化：把分数变成概率，和为1
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = F.dropout(scores, p=self.dropout, training=self.training)

        # 加权求和 ：把概率乘以V，得到每个位置的输出
        output = torch.matmul(scores, xv)

        # 还原形状，把(Batch,Heads,Seq,Dim)变回(Batch,Seq,Dim)
        # contiguous：把张量变成连续的，也就是把张量变成一个一维张量，否则无法执行view
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 投影：把输出变成输出层
        return self.wo(output)
    

# 5.前馈神经网络FeedForward
# 作用：整合信息，增加非线性能力
# LLaMA特色：使用了SwiGLU激活函数，需要三个线性层
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 计算隐藏层维度，通常是输入的4倍
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3) # SwiGLU的特殊调整

        # 调整hidden_dim或者是256的倍数，为了硬件效率
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # 定义三个线性层
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False) # 门控层Gate
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False) # 输出层Down
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False) # 特征层Up

    def forward(self,x):
        # SwiGLU公式：F.silu(w1(x)) * w3(x) -> 再过w2
        # silu就是SiLU激活函数，这里的乘法是逐元素相乘
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        
# 6.Transformer层 TransformerBlock
# 作用：把Attention和FeedForward连接起来，组成一层
class TransformerBlock(nn.Module):
    def __init__(self,layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        # 实例化子模块
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        # 每个子模块前都有一个Norm
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # 残差连接(Residal Connection): x = x + f(x)
        # 先做Norm，再做Attention，结果加回x
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        # 先做Norm，再做FeedForward，结果加回h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
# 7.Transformer模型
# 作用：搭积木，把Embedding，32层Block，输出层组装在一起
class Transformer(nn.Module):
    def __init__(self,params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # 词嵌入层：把Token ID变成向量
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # 堆叠N层TransformerBlock
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        
        # 最终的归一化层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # 输出层：把向量变回词表概率
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 优化点：与计算RoPE旋转矩阵
        # register_buffer告诉pytorch：这是模型的一部分数据，但不是需要更新的参数
        # 这样做的好处是：当你model.to("cuda")时，这些数据会自动跟着去显卡，不用操心
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        # tokens shape: (Batch, Seq_Len)
        bsz, seqlen = tokens.shape

        # 查表：ID变成向量
        h = self.tok_embeddings(tokens)
        
        # 获取对应的RePE 旋转矩阵
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 生成Mask
        # 目标：生成一个上三角全是负无穷的矩阵
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1) # 保留上三角，对角线偏移1

            # 为了处理start_pos(推理时的缓存)，可能需要横向扩展mask
            mask = torch.hstack([torch.zeros((seqlen,start_pos), device=tokens.device),mask]).type_as(h)

        # 一层层流过TransformerBlocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        # 最终归一化
        h = self.norm(h)

        # 映射回词表大小，得到Logits
        output = self.output(h).float()
        return output

# ==========================================
# 增强版验证代码 (Forward + Backward)
# 作用：测试模型能不能跑通，能不能学习（有梯度）。
# ==========================================
if __name__ == "__main__":
    # 1. 检查有没有显卡
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 正在使用计算设备: {device.upper()}")

    # 2. 定义测试参数 
    # (为了演示，我们把模型设得很小，防止你电脑卡死)
    args = ModelArgs(
        dim=512,          # 正常是 4096，这里缩到 512
        n_layers=4,       # 正常是 32，这里只用 4 层
        n_heads=8,        
        vocab_size=5000, 
        max_seq_len=128
    )
    
    # 3. 初始化模型并搬到 GPU
    print("🛠️  正在初始化 LLaMA 架构模型...")
    model = Transformer(args).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数量: {total_params / 1e6:.2f} Million")

    # 4. 构造虚拟数据 (随机生成的数字)
    batch_size = 4
    seq_len = 32
    print(f"📥 输入形状: (Batch={batch_size}, Seq={seq_len})")
    
    inputs = torch.randint(0, args.vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, args.vocab_size, (batch_size, seq_len)).to(device) # 假装这是正确答案

    # ==========================
    # 验证 A: 前向传播 (Forward)
    # ==========================
    print("\n🔄 [Step 1] 测试前向传播 (Forward)...")
    try:
        logits = model(inputs)
        print(f"✅ 前向传播成功！输出形状: {logits.shape}")
        # 检查输出维度是不是 (B, L, Vocab_Size)
        assert logits.shape == (batch_size, seq_len, args.vocab_size)
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        exit()

    # ==========================
    # 验证 B: 反向传播 (梯度检查)
    # ==========================
    print("\n🔄 [Step 2] 测试反向传播 (Backward)...")
    try:
        # 计算 Loss
        # view(-1, ...) 是把数据拉平成一长条，这是 CrossEntropyLoss 要求的格式
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        print(f"📉 当前 Loss: {loss.item():.4f}")
        
        # 反向传播 (求导)
        loss.backward()
        
        # 检查第一层 (Embedding) 有没有收到梯度
        # 如果 grad 不是 None 且 norm > 0，说明网络是通的！
        grad_norm = model.tok_embeddings.weight.grad.norm().item()
        print(f"✅ 反向传播成功！梯度已生成。")
        print(f"🔍 Token Embedding 层梯度范数: {grad_norm:.4f}")
        
        if grad_norm > 0:
            print("\n🎉 恭喜！模型复现成功，且具备学习能力！")
        else:
            print("\n⚠️ 警告：梯度为 0，可能存在断链。")
            
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")