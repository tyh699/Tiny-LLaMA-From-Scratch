import torch
from torch.utils.data import Dataset
import json
import sentencepiece as spm

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_length=512):
        self.max_length = max_length
        # 加载分词器
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = 0 

        # 加载数据
        print(f"正在加载 SFT 数据: {data_path} ...")
        self.data_list = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data_list.append(json.loads(line))
        print(f"SFT 数据加载完毕，共 {len(self.data_list)} 条。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        
        # ---------------------------------------------------------
        # 【面试考点 1】对话模板构建
        # ---------------------------------------------------------
        # 我们的目标格式：[BOS] User: 问题 \n AI: 回答 [EOS]
        
        # 1. 处理 Prompt (提问部分)
        # 这里的 "User:" 和 "AI:" 是硬编码的提示词，帮助模型区分角色
        prompt_text = f"User: {item.get('instruction', '')} {item.get('input', '')}\nAI: "
        prompt_ids = self.sp.encode_as_ids(prompt_text)
        prompt_ids = [self.bos_id] + prompt_ids # 加上开头符
        
        # 2. 处理 Answer (回答部分)
        answer_text = item.get('output', "")
        answer_ids = self.sp.encode_as_ids(answer_text)
        answer_ids = answer_ids + [self.eos_id] # 加上结束符
        
        # 3. 拼接 Input (模型看到的完整输入)
        input_ids = prompt_ids + answer_ids
        
        # ---------------------------------------------------------
        # 【面试考点 2】Loss Masking (核心中的核心)
        # ---------------------------------------------------------
        # 我们只希望模型学习“如何回答”，而不学习“如何提问”。
        # 所以，Prompt 部分的 Label 设为 -100 (PyTorch 会自动忽略这个值)
        # Answer 部分的 Label 设为真实的 token ID
        labels = [-100] * len(prompt_ids) + answer_ids
        
        # 4. 截断与填充 (Padding)
        # 如果太长，就截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # 如果太短，就补 0 (Padding)
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.pad_id] * padding_len
            # Padding 部分也不算 Loss，所以也设为 -100
            labels = labels + [-100] * padding_len 
            
        # 转为 Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return input_ids, labels