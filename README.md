# BasicGPT

A from-scratch GPT implementation for learning. Built alongside Cursor.
NOT a `build me a GPT, no mistakes` codebase, more of piece by piece
 `lets add a class called datasetprep to share and prepare each invidual dataset`
Supports single and multi-GPU training via Hugging Face Accelerate.

## Quick Start

```bash
pip install -r requirements.txt
python train.py                    # Single GPU
accelerate launch train.py         # Multi-GPU
```

---

## Scripts

### Training & Generation

| Script | Usage |
|--------|-------|
| `train.py` | `accelerate launch train.py` |
| `generate.py` | `python generate.py <checkpoint> --prompt "text"` |
| `evals.py` | `python evals.py <checkpoint>` |
| `prepare_data.py` | `python prepare_data.py` (test data loading) |

### `generate.py` options
```bash
--prompt "text"      # Required
--max_tokens 200     # Max tokens to generate
--temperature 0.8    # 0=greedy, 1=random
--top_k 50          # Top-k sampling
--top_p 0.9         # Nucleus sampling
```

---

## Vast.ai / Remote Training

### Setup & Train (on remote)

```bash
./start.sh              # Install deps + detect GPUs + train
./start.sh --skip-deps  # Skip pip install
./start.sh --dry-run    # Setup only, no training
```

### Checkpoint Sync (on local machine)

```bash
# First, edit scripts with your remote details:
# REMOTE_HOST, REMOTE_PORT in both files

# Test connectivity before training
./scripts/test_remote_sync.sh

# Sync checkpoints during training
./scripts/sync_checkpoints.sh              # One-time
./scripts/sync_checkpoints.sh --watch      # Every 60s
./scripts/sync_checkpoints.sh --delete     # Delete after download
./scripts/sync_checkpoints.sh --watch --delete  # Recommended
```

---

## Accelerate Commands

```bash
accelerate launch train.py                      # Auto-detect GPUs
accelerate launch --num_processes=8 train.py    # Specific count
accelerate config                               # First-time setup
```

---

## Configuration

All in `config.py`:

```python
# Datasets (DataConfig)
current_datasets: [FINEWEB, C4]
dataset_probabilities: [0.6, 0.4]  # Must match dataset count
max_samples: 10000000

# Model (GPTConfig)
d_model: 256
n_layers: 16
n_heads: 8
max_length: 1024

# Training (TrainingConfig)  
batch_size: 32
gradient_accumulation_steps: 4
epochs: 3
```

Effective batch = `batch_size × grad_accum × num_gpus`

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `scripts/calc_params.py` | Model parameter calculator |
| `scripts/calc_lr_config.py` | LR schedule calculator |
| `scripts/sync_checkpoints.sh` | Download checkpoints from remote |
| `scripts/test_remote_sync.sh` | Test remote connectivity |

---

## License

MIT
