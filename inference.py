import torch
import sentencepiece as spm
from model import Transformer, ModelArgs
import os
import time

def main():
    # ========================================================
    # 1. 基础配置
    # ========================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 确保加载的是最新训练好的 SFT 模型
    checkpoint_path = "./checkpoints_sft/model_sft_final.pth" 
    tokenizer_path = "./tokenizer.model"
    
    # 模型参数 (必须与训练时一致)
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=32000,
        max_seq_len=512
    )

    # ========================================================
    # 2. 加载模型与分词器
    # ========================================================
    print(f"🚀 正在加载模型: {checkpoint_path} ...")
    if not os.path.exists(checkpoint_path):
        print("❌ 错误：找不到模型文件！请检查路径。")
        return

    # 初始化架构
    model = Transformer(args).to(device)
    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    # 【重要】切换到评估模式 (关闭 Dropout)
    model.eval()
    print("✅ 模型加载完成！")

    # 加载分词器
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    print("✅ 分词器加载完成！")

    # ========================================================
    # 3. 核心生成函数 (支持 KV Cache)
    # ========================================================
    def chat(prompt, temperature=0.8, max_len=100):
        """
        prompt: 用户输入的问题
        temperature: 采样温度 (越高越发散，0为贪婪搜索)
        max_len: 最大生成长度
        """
        
        # --- A. 构建对话模板 ---
        # SFT 模型需要这种特定的格式才能听懂
        formatted_prompt = f"User: {prompt}\nAI: "
        
        # 编码并添加 BOS (Start Token)
        input_ids = sp.encode_as_ids(formatted_prompt)
        input_ids = [sp.bos_id()] + input_ids
        
        # 转 Tensor，并搬运到 GPU
        # x 初始形状: (1, seq_len)
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # 打印 AI 开头，准备流式输出
        print("AI: ", end="", flush=True)

        # --- B. 初始化 KV Cache 变量 ---
        # kv_caches 初始为 None，模型内部会自动创建
        kv_caches = None 
        # start_pos 记录当前输入的起始位置 (用于 RoPE 位置编码)
        start_pos = 0 
        
        start_time = time.time()
        new_tokens_count = 0

        # --- C. 生成循环 ---
        with torch.no_grad(): # 推理模式不需要算梯度
            for i in range(max_len):
                
                # [核心逻辑] 根据是否是第一步，决定如何传参
                if i == 0:
                    # === Prefill 阶段 (预填充) ===
                    # 第一步把整个 Prompt 喂进去
                    # start_pos=0，模型会计算所有 token 的 KV 并存入 cache
                    logits, kv_caches = model(x, start_pos=0, kv_caches=None)
                    
                    # 更新 start_pos：现在的长度就是下一次的起点
                    start_pos = x.shape[1] 
                else:
                    # === Decode 阶段 (解码) ===
                    # 后续步骤只喂这一个新生成的字 (last token)
                    # 此时 x 的形状必须是 (1, 1)
                    # start_pos 每次 +1
                    logits, kv_caches = model(x, start_pos=start_pos, kv_caches=kv_caches)
                    start_pos += 1

                # 取最后一个时间步的预测结果
                last_token_logits = logits[0, -1, :]

                # [采样逻辑]
                if temperature < 1e-5:
                    # 贪婪搜索 (Argmax): 总是选概率最大的
                    next_token = torch.argmax(last_token_logits).item()
                else:
                    # 随机采样 (Multinomial): 根据概率分布抽签
                    probs = torch.softmax(last_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                # 遇到结束符 (EOS) 停止
                if next_token == sp.eos_id():
                    break
                
                # 解码并打印
                word = sp.decode_ids([next_token])
                print(word, end="", flush=True)
                new_tokens_count += 1

                # [重要] 准备下一次输入
                # 因为用了 KV Cache，我们只需要传这一个新 token
                # 形状保持 (1, 1)
                x = torch.tensor([[next_token]], dtype=torch.long, device=device)
        
        # 打印统计信息
        end_time = time.time()
        speed = new_tokens_count / (end_time - start_time)
        print(f"\n[Speed: {speed:.2f} token/s]\n")

    # ========================================================
    # 4. 交互式循环 (Control Loop)
    # ========================================================
    print("\n💬 欢迎使用 HappyLLM 对话终端！(输入 'q' 或 'exit' 退出)")
    print("-" * 50)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\nUser: ")
            
            # 退出指令
            if user_input.lower() in ["q", "exit", "quit"]:
                print("👋 再见！")
                break
            
            # 忽略空输入
            if not user_input.strip():
                continue
            
            # 调用聊天函数
            # 这里温度设为 0.8，让它稍微有点创造力
            chat(user_input, temperature=0.8)
            
        except KeyboardInterrupt:
            print("\n👋 用户强制退出。")
            break

if __name__ == "__main__":
    main()