# 🦙 Tiny-LLaMA-From-Scratch (HappyLLM Implementation)

> **本项目是一个基于 PyTorch 原生代码的大语言模型复现工程。**
> 旨在从零开始（From Scratch）构建一个架构对齐 LLaMA 的 Transformer 模型，并在单卡 RTX 5070 (12GB) 上完成了从 Tokenizer 训练、预训练 (Pretrain) 到指令微调 (SFT) 的全流程闭环。

<div align="center">

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange.svg)

</div>

## ⚠️ 项目性质说明 (Disclaimer)

> **本项目为个人学习性质的复现工程 (Educational Purpose)。**
>
> 为了快速验证代码管线与架构的正确性，模型仅使用 **10,000 条 (10k)** 样本进行训练。因此，模型**不具备**实际的对话智能或逻辑推理能力（可能会出现复读、逻辑不通等现象）。
>
> **本项目的核心价值在于：** 跑通大模型从 0 到 1 的完整代码流程，深入理解 Transformer 底层细节与训练机制。

## 🌟 项目亮点 (Key Features)

本项目不依赖 `transformers` 高层库的现成模型接口，而是通过 `torch.nn` 手写实现了以下核心组件，用以深入理解 LLM 底层原理：

* **核心架构 (Model Architecture)**:
    * **RMSNorm**: 相比 LayerNorm 计算更高效的归一化层。
    * **SwiGLU**: LLaMA 标志性的激活函数，增强模型的非线性表达能力。
    * **RoPE (Rotary Positional Embeddings)**: 实现了旋转位置编码，通过复数旋转矩阵注入位置信息，更好地处理长文本序列。
    * **GQA (Grouped Query Attention)**: 实现了分组查询注意力机制（逻辑支持），为 KV Cache 显存优化打下基础。
* **训练管线 (Training Pipeline)**:
    * **Tokenizer**: 基于 SentencePiece 训练了专属的中文分词器 (BPE)。
    * **Pretrain**: 实现了标准的 Next Token Prediction 预训练任务。
    * **SFT (Supervised Fine-Tuning)**: 实现了带有 **Loss Masking** 机制的指令微调，通过 `ignore_index=-100` 屏蔽 Prompt 部分的梯度，强制模型专注于 Answer 生成。

## 📂 目录结构 (Directory Structure)

```text
.
├── checkpoints/          # (已忽略) 预训练模型权重存档
├── checkpoints_sft/      # (已忽略) SFT 微调后模型权重存档
├── data/                 # (已忽略) 存放 jsonl 数据集
├── model.py              # 【核心】Transformer、RMSNorm、RoPE、Attention 架构实现
├── dataset.py            # 预训练数据处理 (Padding, Tokenization)
├── dataset_sft.py        # SFT 数据处理 (Prompt Template, Loss Masking)
├── train.py              # 预训练脚本
├── train_sft.py          # SFT 微调脚本 (加载预训练权重 -> 微调)
├── inference.py          # 基础模型推理脚本
├── inference_sft.py      # SFT 对话模型推理脚本 (含对话模板)
├── train_tokenizer.py    # 分词器训练脚本
├── tokenizer.model       # 训练好的分词器二进制文件
├── requirements.txt      # 项目依赖库列表
└── README.md             # 项目说明文档
``` 

## 🛠️ 快速开始 (Quick Start)

### 1. 环境准备

建议创建独立的 Conda 环境，并安装依赖：

Bash

```
pip install -r requirements.txt
```

### 2. 数据准备与分词器训练

下载测试数据并训练 SentencePiece 分词器：

Bash

```
# 1. 下载少量测试数据 (mini_data.jsonl)
python download_mini_data.py

# 2. 训练 Tokenizer (生成 tokenizer.model)
python train_tokenizer.py
```

### 3. 阶段一：预训练 (Pretraining)

从零随机初始化模型，学习语言的基本概率分布：

Bash

```
python train.py
```

- *Output*: 模型权重将保存在 `checkpoints/` 目录下。

### 4. 阶段二：指令微调 (SFT)

加载预训练好的权重，使用对话数据进行微调，使模型具备指令跟随能力：

Bash

```
python train_sft.py
```

- *Key Tech*: 此阶段应用了 Mask 机制，不计算 "User" 提问部分的 Loss。

### 5. 推理与对话 (Inference)

与微调后的模型进行对话测试：

Bash

```
python inference_sft.py
```

## 📊 实验结果 (Results)

- **硬件环境**: NVIDIA RTX 5070 (12GB VRAM)
- **训练效率**: 在单卡环境下成功跑通全流程，验证了小参数量模型在特定硬件下的可训练性。
- **SFT 效果**: 模型能够严格遵循 `User: <query> \n AI: <response>` 的对话模板格式进行回复，验证了 SFT 数据构造 pipeline 和 Mask 机制的正确性。

## 🔗 参考资料 (References)

- **Paper**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Guide**: [Datawhale HappyLLM Project](https://github.com/datawhalechina/happy-llm)
- **Guide**：https://datawhalechina.github.io/happy-llm

## 🙏 致谢 (Acknowledgements)

感谢以下项目和社区提供的学习资源：

- 感谢 **[Datawhale](https://github.com/datawhalechina)** 提供的开源教程与社区支持。
- 感谢 **Meta AI** 开源的 LLaMA 架构设计思路。
- 感谢 **PyTorch** 与 **SentencePiece** 提供的底层工具库支持。

------

*Created by Tang Yuanhang for Learning Purpose.*

