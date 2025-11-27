# BasicGPT

A from-scratch GPT implementation for learning. Supports pretraining, supervised fine-tuning (SFT), and reinforcement learning (DPO).

## Project Structure

```
BasicGPT/
├── model/                    # GPT architecture
│   ├── gpt.py               # Core GPT model class
│   └── config.py            # Model configuration
│
├── data/                     # Data handling
│   ├── tokenizer.py         # Tiktoken wrapper
│   ├── datasets.py          # Dataset loading & preprocessing
│   └── loaders/             # Individual dataset implementations
│       ├── datasetprep.py   # Base class
│       ├── finewebdataset.py
│       ├── c4dataset.py
│       └── synthdataset.py
│
├── training/                 # Training pipelines
│   ├── pretrain/
│   │   └── train.py         # Pretraining script
│   ├── finetune/
│   │   ├── config.py        # SFT configuration
│   │   └── sft_trainer.py   # SFT trainer (uses GPT-2)
│   └── rl/
│       ├── config.py        # DPO/PPO configuration
│       └── dpo_trainer.py   # DPO trainer
│
├── evals/                    # Evaluation & generation
│   ├── evaluate.py          # Model evaluation
│   └── generate.py          # Text generation
│
├── utils/                    # Utilities
│   ├── device.py            # Device detection
│   └── checkpoints.py       # Checkpoint handling
│
├── config.py                 # Shared configs (Training, Data, etc.)
├── enums.py                  # Dataset enums
├── learning_rate.py          # LR schedule
├── gpt.py                    # Training functions (train_epoch, evaluate)
│
├── train.py                  # Entry point (backward compat)
├── run_sft.py               # SFT entry point
├── run_dpo.py               # DPO entry point
│
├── start.sh                  # Vast.ai launcher
└── scripts/                  # Shell helpers
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

### 1. Pretraining
```bash
./start.sh                           # On Vast.ai
accelerate launch train.py           # Local multi-GPU
python train.py                       # Local single GPU
```

### 2. Supervised Fine-Tuning (GPT-2)
```bash
python run_sft.py                     # Fine-tune GPT-2 on Alpaca
python run_sft.py --model gpt2-medium --epochs 3
```

### 3. DPO (After SFT)
```bash
python run_dpo.py --model ./checkpoints/sft/epoch_3
```

### 4. Evaluation & Generation
```bash
python -m evals.evaluate ./checkpoints/your-checkpoint/
python -m evals.generate ./checkpoints/your-checkpoint/ --prompt "Hello"
```

---

## Training Stages

| Stage | Script | Base Model | Data |
|-------|--------|------------|------|
| **Pretrain** | `start.sh` / `train.py` | Random init | FineWeb, C4 |
| **SFT** | `run_sft.py` | GPT-2 | Alpaca instructions |
| **DPO** | `run_dpo.py` | SFT model | HH-RLHF preferences |

---

## Configuration

### Model (`model/config.py`)
```python
GPTConfig(
    d_model=256,
    n_layers=16,
    n_heads=8,
    max_length=1024,
)
```

### Pretraining (`config.py`)
```python
DataConfig(
    current_datasets=[FINEWEB, C4],
    dataset_probabilities=[0.6, 0.4],
)

TrainingConfig(
    batch_size=32,
    gradient_accumulation_steps=4,
    epochs=3,
)
```

### SFT/DPO (in `training/*/config.py`)
```python
SFTConfig(model_name="gpt2", dataset_name="tatsu-lab/alpaca")
DPOConfig(model_name="path/to/sft", beta=0.1)
```

---

## Vast.ai Deployment

```bash
# Upload code
./scripts/upload.sh

# On remote
./start.sh

# Sync checkpoints (locally)
./scripts/sync_checkpoints.sh --watch
```

---

## License

MIT
