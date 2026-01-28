import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# 导入你的模块
from model import Transformer, ModelArgs
from dataset_sft import SFTDataset # 注意这里导入的是新的 Dataset

def train_sft():
    # --- 1. 配置参数 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4        # 显存够的话可以调大
    max_seq_len = 512
    learning_rate = 1e-5  # 【注意】SFT 的学习率通常比预训练要小 (1/10 左右)
    epochs = 2            # 微调通常只需要很少的轮数
    
    # 路径配置
    pretrain_ckpt = "./checkpoints/model_final.pth" # 预训练好的模型
    save_dir = "./checkpoints_sft"                  # SFT 模型保存位置
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 初始化模型 ---
    print("正在初始化模型架构...")
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=32000,
        max_seq_len=max_seq_len
    )
    model = Transformer(args).to(device)

    # ---------------------------------------------------------
    # 【面试考点 3】加载预训练权重 (Transfer Learning)
    # ---------------------------------------------------------
    print(f"正在加载预训练权重: {pretrain_ckpt} ...")
    if os.path.exists(pretrain_ckpt):
        state_dict = torch.load(pretrain_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 权重加载成功！开始在巨人的肩膀上微调。")
    else:
        print("❌ 警告：没找到预训练权重！将从头开始训练（这就不叫 SFT 了）。")

    # --- 3. 准备数据 ---
    dataset = SFTDataset(
        data_path="./data/mini_data.jsonl",
        tokenizer_path="./tokenizer.model",
        max_length=max_seq_len
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 4. 优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 5. 训练循环 ---
    model.train()
    print(f"开始 SFT 训练！设备: {device}, 数据量: {len(dataset)}")
    
    step = 0
    start_time = time.time()

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # 前向传播
            logits, _ = model(x)

            # 计算 Loss
            # 注意：dataset_sft.py 里已经把 Prompt 部分的 label 设为 -100 了
            # 但我们还是要做 shift (错位)，因为是预测下一个词
            
            # [Input]:  A B C D
            # [Label]: -1 -1 C D  (假设 A B 是 Prompt)
            
            # Shift 后：
            # Preds (logits): 预测 B, 预测 C, 预测 D
            # Targets (y):    -100,  C,      D
            
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, args.vocab_size),
                y[:, 1:].reshape(-1),
                ignore_index=-100 # 【关键】忽略掉 -100 的部分
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            
    # 保存 SFT 后的模型
    final_path = os.path.join(save_dir, "model_sft_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"SFT 训练结束！最终模型保存至: {final_path}")

if __name__ == "__main__":
    train_sft()