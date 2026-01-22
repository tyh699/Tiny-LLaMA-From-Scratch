import json
import sentencepiece as spm

def train_tokenizer():
    # 1. 第一步：把 jsonl 里的文本提取出来，存成一个临时的 txt 文件
    # 因为 SentencePiece 训练需要直接读取纯文本文件
    input_file = "./data/mini_data.jsonl"
    txt_file = "./data/corpus.txt"
    model_prefix = "tokenizer"  # 训练模型的前缀名称

    print(f"正在将 {input_file} 转换为纯文本...")
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(txt_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            # 我们把 instruction, input, output 全部拼在一起作为训练语料
            text = data.get("instruction", "") + data.get("input", "") + data.get("output", "")
            f_out.write(text + "\n")
    
    # 2. 第二步：调用 SentencePiece 训练
    # vocab_size: 词表大小，通常小模型设为 32000 或 64000
    # model_type: 使用 BPE 算法
    print("开始训练 Tokenizer (可能需要几分钟)...")
    spm.SentencePieceTrainer.train(
        input=txt_file,
        model_prefix=model_prefix,
        vocab_size=32000,  # 如果数据量太少，这里可能报错，可以改小一点比如 8000
        user_defined_symbols=['<pad>', '<bos>', '<eos>'], # 定义特殊符号
        model_type="bpe",
        byte_fallback=True # 支持生僻字
    )
    print(f"训练完成！生成了 {model_prefix}.model 和 {model_prefix}.vocab")

if __name__ == "__main__":
    train_tokenizer()