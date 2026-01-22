import torch
from torch.utils.data import Dataset, DataLoader
import json
import sentencepiece as spm
import numpy as np

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_length=512):
        self.max_length = max_length
        
        # 1. 加载 Tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        # 获取特殊 token 的 ID
        self.bos_id = self.sp.bos_id() # Begin of Sentence
        self.eos_id = self.sp.eos_id() # End of Sentence
        self.pad_id = self.sp.pad_id() # Padding (通常是 0 或 -1，这里 SentencePiece 默认可能有定义)
        # 如果 tokenizer 没定义 pad_id，我们就手动指定一个（比如 0）
        if self.pad_id == -1:
            self.pad_id = 0

        # 2. 加载数据
        print(f"正在加载数据: {data_path} ...")
        self.data_list = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data_list.append(json.loads(line))
        print(f"数据加载完毕，共 {len(self.data_list)} 条。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 1. 获取文本
        item = self.data_list[index]
        # 把 instruction, input, output 拼成一整段话进行预训练
        # 格式：Instruction + Input + Output
        text = item.get("instruction", "") + item.get("input", "") + item.get("output", "")
        
        # 2. 文本转 ID (Tokenize)
        input_ids = self.sp.encode_as_ids(text)
        
        # 3. 加上 BOS 和 EOS
        # [BOS] + 文本 + [EOS]
        input_ids = [self.bos_id] + input_ids + [self.eos_id]
        
        # 4. 截断与填充 (Padding & Truncation)
        # 这一步非常重要，必须保证 output 的长度严格等于 max_length
        
        # 情况 A: 句子太长 -> 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # 情况 B: 句子太短 -> 填充
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.pad_id] * padding_len
            
        # 5. 转为 Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 训练时，Input 和 Target 通常是一样的
        # 模型内部会负责把 Input 往后移一位来和 Target 对比
        x = input_ids
        y = input_ids # Label
        
        return x, y

# ==========================================
# 单元测试代码 (一定要保留，用来验证数据格式对不对)
# ==========================================
if __name__ == "__main__":
    # 假设你的文件路径如下，如果不对请修改
    data_path = "./data/mini_data.jsonl"
    tokenizer_path = "./tokenizer.model"
    
    # 初始化 Dataset
    ds = PretrainDataset(data_path, tokenizer_path, max_length=64)
    
    # 看看第一条数据长什么样
    x, y = ds[0]
    print("------------------------------------------------")
    print(f"数据总数: {len(ds)}")
    print(f"Sample Input Shape: {x.shape}")
    print(f"Sample Input IDs: {x.tolist()}")
    print("------------------------------------------------")
    
    # 测试 DataLoader (模拟训练时的批次读取)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch_x, batch_y in loader:
        print(f"Batch X Shape: {batch_x.shape}") # 应该是 [4, 64]
        print(f"Batch Y Shape: {batch_y.shape}")
        break
    print("✅ 数据管道 dataset.py 测试通过！")