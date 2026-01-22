import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# 导入我们之前写好的模块
from model import Transformer, ModelArgs
from dataset import PretrainDataset

def train():
    # --- 1. 配置参数 (根据你的显存调整) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 显存优化策略：
    # 如果显存爆了，把 batch_size 调小 (比如 4 -> 2)
    # 如果想跑快点，把 batch_size 调大
    batch_size = 4  
    max_seq_len = 512 
    learning_rate = 3e-4 # 学习率，常见值 3e-4
    epochs = 1           # 演示项目跑 1 个 epoch 就够了
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 初始化模型 ---
    print("正在初始化模型...")
    args = ModelArgs(
        dim=512,            # 和 model.py 保持一致
        n_layers=8,
        n_heads=8,
        vocab_size=32000,   # 必须和 tokenizer.vocab 大小一致
        max_seq_len=max_seq_len
    )
    model = Transformer(args).to(device)
    
    # --- 3. 准备数据 ---
    dataset = PretrainDataset(
        data_path="./data/mini_data.jsonl",
        tokenizer_path="./tokenizer.model",
        max_length=max_seq_len
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # --- 4. 优化器 (面试必看) ---
    # AdamW 是目前大模型训练最主流的优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 5. 训练循环 ---
    model.train() # 开启训练模式 (启用 Dropout)
    print(f"开始训练！设备: {device}, 数据量: {len(dataset)}")
    
    step = 0
    total_steps = len(dataloader) * epochs
    start_time = time.time()

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # 【面试必看 1】前向传播
            # 这里的 logits 形状是 (Batch, Seq_Len, Vocab_Size)
            logits = model(x)

            # 【面试必看 2】计算 Loss
            # 大模型的训练本质就是分类问题：预测下一个词的概率
            # Shift 错位操作：
            # 预测时：我们用第 t 个词去预测第 t+1 个词
            # 所以 logits 取 [:, :-1, :] (去掉最后一个预测，因为没有答案对应它)
            # targets 取 [:, 1:]      (去掉第一个输入，因为它不是任何人的预测目标)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, args.vocab_size), # 展平
                y[:, 1:].reshape(-1),                           # 展平
                ignore_index=0 # 忽略 padding (pad_id=0)
            )

            # 【面试必看 3】反向传播与参数更新
            optimizer.zero_grad() # 清空上一步的梯度
            loss.backward()       # 计算新梯度
            optimizer.step()      # 更新参数

            step += 1
            if step % 10 == 0:
                # 打印日志
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{total_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            
            # 每 100 步保存一次模型
            if step % 100 == 0:
                checkpoint_path = os.path.join(save_dir, f"model_step_{step}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"模型已保存: {checkpoint_path}")

    # 保存最终模型
    final_path = os.path.join(save_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"训练结束！最终模型保存至: {final_path}")

if __name__ == "__main__":
    train()