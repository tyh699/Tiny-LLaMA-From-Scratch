import torch
import sentencepiece as spm
from model import Transformer, ModelArgs
import os

def generate():
    # --- 1. 基础配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "./checkpoints/model_final.pth"
    tokenizer_path = "./tokenizer.model"
    
    # 必须和 train.py 里的参数完全一致！
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=32000,
        max_seq_len=512
    )

    # --- 2. 加载模型 ---
    print(f"正在加载模型: {checkpoint_path} ...")
    if not os.path.exists(checkpoint_path):
        print("错误：找不到模型文件！请检查路径。")
        return

    model = Transformer(args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # 切换到评估模式

    # --- 3. 加载分词器 ---
    print(f"正在加载分词器: {tokenizer_path} ...")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # --- 4. 定义交互函数 ---
    # 简单的贪婪搜索 (Greedy Search)
    def chat(prompt):
        input_ids = sp.encode_as_ids(prompt)
        input_ids = [sp.bos_id()] + input_ids
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        print(f"\n用户: {prompt}")
        print("AI: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(50): # 最多生成 50 个字
                logits = model(x)
                # 取最后一个 token 的预测结果
                last_token_logits = logits[0, -1, :]
                # 选概率最大的那个字 (Argmax)
                next_token = torch.argmax(last_token_logits).item()
                
                # 遇到结束符就停
                if next_token == sp.eos_id():
                    break
                
                # 解码并打印
                word = sp.decode_ids([next_token])
                print(word, end="", flush=True)
                
                # 把生成的字加回去，继续预测下一个
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
        print() # 换行

    # --- 5. 开始测试 ---
    print("\n=== 模型测试 (输入 'q' 退出) ===")
    test_sentences = ["你好", "人工智能", "北京邮电大学"]
    for s in test_sentences:
        chat(s)

    while True:
        user_input = input("\n请输入提示词: ")
        if user_input.lower() == 'q':
            break
        chat(user_input)

if __name__ == "__main__":
    generate()