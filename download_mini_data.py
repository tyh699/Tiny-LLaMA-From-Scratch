# download_mini_data.py
import json
from datasets import load_dataset
import os

# 目标保存路径
save_path = "./data/mini_data.jsonl"
os.makedirs("./data", exist_ok=True)

print("正在流式加载 BelleGroup/train_0.5M_CN 数据集...")
# 使用 streaming=True，这样不需要下载整个数据集，可以像水流一样通过
# 这里我们使用 BelleGroup 的中文指令数据集，也是 HappyLLM 提到的 SFT 常用数据
# 如果想复现 Pretrain，我们可以把指令部分的 Input/Output 拼起来当纯文本用
dataset = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)

print("开始抽取前 10,000 条数据...")
data_list = []
count = 0

for item in dataset:
    # item 也是一个字典，通常包含 'instruction', 'input', 'output'
    # 我们将其保存下来
    data_list.append(item)
    count += 1
    if count % 1000 == 0:
        print(f"已下载 {count} 条...")
    if count >= 10000:
        break

# 写入本地 jsonl 文件
print(f"正在写入 {save_path} ...")
with open(save_path, "w", encoding="utf-8") as f:
    for line in data_list:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print("完成！数据已就绪。")