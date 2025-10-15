# TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning

Implementation of "TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning".

This repo fine-tunes an LLM to be more truth-seeking using a GRPO-style trainer (Group Relative Policy Optimization) with task-specific reward signals and evaluation hooks.

📄 **Paper:** [https://arxiv.org/pdf/2509.25760](https://arxiv.org/pdf/2509.25760)

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install packaging
pip install ninja
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install vllm
pip install onnxruntime-gpu
pip install datasets
pip install transformers
pip install python-dotenv
pip install uuid
pip install openai
pip install math-verify
pip install jsonlines
pip install tqdm
pip install pandas
pip install wandb

cd src
pip install -e . --user
pip install -U deepspeed
cd ..
```

### 2. 🧠 Running the Trainer
Once dependencies are installed, launch training with:
```bash
bash ./src/scripts/grpo.sh
```


## 🧩 TODO
- More detailed training dynamics will be provided soon
- Momprehensive evaluation results will also be provided