import torch
import sentencepiece as spm
from model import Transformer, ModelArgs
import os

def generate_sft():
    # --- 1. 配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 【注意】这里加载的是 SFT 后的模型权重
    checkpoint_path = "./checkpoints_sft/model_sft_final.pth"
    tokenizer_path = "./tokenizer.model"
    
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=32000,
        max_seq_len=512
    )

    # --- 2. 加载模型 ---
    print(f"正在加载 SFT 模型: {checkpoint_path} ...")
    if not os.path.exists(checkpoint_path):
        print("错误：找不到模型文件！请检查路径。")
        return

    model = Transformer(args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 3. 加载分词器 ---
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # --- 4. 定义对话函数 ---
    def chat(prompt):
        # ==========================================
        # 【重点】SFT 推理必须加上对话模板！
        # 训练时模型看的是 "User: xxx \n AI: yyy"
        # 所以推理时，我们发给它的也必须是 "User: xxx \n AI: "
        # ==========================================
        formatted_prompt = f"User: {prompt}\nAI: "
        
        input_ids = sp.encode_as_ids(formatted_prompt)
        input_ids = [sp.bos_id()] + input_ids
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        print(f"\n用户: {prompt}")
        print("AI: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(50): 
                logits = model(x)
                last_token_logits = logits[0, -1, :]
                
                # 稍微加一点温度，让它说话生动点，不要死板
                temperature = 0.8
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == sp.eos_id():
                    break
                
                word = sp.decode_ids([next_token])
                print(word, end="", flush=True)
                
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
        print()

    # --- 5. 开始测试 ---
    print("\n=== SFT 模型测试 (已启用对话模板) ===")
    
    # 测试几个典型问题
    test_queries = [
        "你好",
        "介绍一下你自己",
        "北京在哪里",
        "1+1等于几"
    ]
    
    for q in test_queries:
        chat(q)

    while True:
        user_input = input("\n请输入问题 (q退出): ")
        if user_input.lower() == 'q':
            break
        chat(user_input)

if __name__ == "__main__":
    generate_sft()