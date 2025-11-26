# BasicGPT

A from-scratch GPT implementation for learning and experimentation. Supports single and multi-GPU training via Hugging Face Accelerate.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train (single GPU)
python train.py

# Train (multi-GPU)
accelerate launch train.py
```

---

## Project Structure

```
BasicGPT/
├── train.py              # Main training script (supports multi-GPU)
├── generate.py           # Text generation from checkpoints
├── evals.py              # Model evaluation (perplexity, accuracy)
├── gpt.py                # GPT model architecture
├── config.py             # All configuration classes
├── prepare_data.py       # Dataset loading & preprocessing
├── tokenizer.py          # Tokenizer wrapper (tiktoken)
├── device.py             # Device detection utilities
├── learning_rate.py      # Learning rate schedule
├── enums.py              # Dataset enums
├── utils.py              # Utility functions
├── dataset_loaders/      # Dataset-specific loaders
│   ├── synthdataset.py   # SYNTH dataset
│   ├── finewebdataset.py # FineWeb dataset
│   ├── c4dataset.py      # C4 dataset
│   └── ...
├── scripts/              # Helper scripts
│   ├── calc_params.py    # Calculate model parameters
│   ├── calc_lr_config.py # Calculate LR schedule
│   └── vast.sh           # Vast.ai deployment
├── start.sh              # Setup & run script
└── checkpoints/          # Saved model checkpoints
```

---

## Core Scripts

### `train.py` - Training

The main training script with multi-GPU support via Accelerate.

```bash
# Single GPU (local development)
python train.py

# Multi-GPU via Accelerate
accelerate launch train.py

# Specify number of GPUs
accelerate launch --num_processes=4 train.py
accelerate launch --num_processes=8 train.py
```

**What it does:**
- Loads datasets (SYNTH + FineWeb by default, 70/30 split)
- Creates GPT model
- Trains with gradient accumulation
- Saves checkpoints periodically and at epoch end
- Runs validation during training

### `generate.py` - Text Generation

Generate text from a trained checkpoint.

```bash
# Basic usage
python generate.py <checkpoint_path> --prompt "Once upon a time"

# With options
python generate.py ./checkpoints/data-5m-batch-50000-11-26-2025-14-30/checkpoint.pt \
    --prompt "The meaning of life is" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `checkpoint_path` | Path to checkpoint file or folder | Required |
| `--prompt` | Text prompt to continue | Required |
| `--max_tokens` | Maximum tokens to generate | 200 |
| `--temperature` | Sampling temperature (0=greedy, 1=random) | 0.8 |
| `--top_k` | Top-k sampling (0=disabled) | 50 |
| `--top_p` | Nucleus sampling threshold | 0.9 |

### `evals.py` - Evaluation

Evaluate model performance on validation data.

```bash
# Evaluate a checkpoint
python evals.py <checkpoint_path>

# Example
python evals.py ./checkpoints/data-5m-batch-156250-11-26-2025-18-00/
```

**Metrics reported:**
- Perplexity
- Cross-entropy loss
- Token prediction accuracy
- Sample generations

### `prepare_data.py` - Data Preparation

Test data loading and tokenization.

```bash
# Run data loading tests
python prepare_data.py
```

---

## Accelerate Commands

Accelerate handles distributed training automatically.

### First-Time Setup

```bash
# Interactive configuration (recommended)
accelerate config
```

This creates `~/.cache/huggingface/accelerate/default_config.yaml`.

### Common Launch Commands

```bash
# Single GPU (same as python train.py)
accelerate launch train.py

# Multi-GPU (auto-detect all GPUs)
accelerate launch --multi_gpu train.py

# Specific number of GPUs
accelerate launch --num_processes=4 train.py
accelerate launch --num_processes=8 train.py

# Specific GPU selection
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 train.py

# With mixed precision override
accelerate launch --mixed_precision=bf16 train.py
```

### Sample Accelerate Config (8x A100)

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 8
mixed_precision: bf16
```

---

## Configuration

All settings are in `config.py`. Key configurations:

### Model Architecture (`GPTConfig`)

```python
vocab_size: int = 100277    # Vocabulary size (tiktoken cl100k_base)
d_model: int = 256          # Embedding dimension
n_heads: int = 8            # Attention heads
n_layers: int = 16          # Transformer layers
max_length: int = 1024      # Context window
dropout: float = 0.1        # Dropout rate
```

### Training (`TrainingConfig`)

```python
batch_size: int = 32                    # Per-GPU batch size
gradient_accumulation_steps: int = 4    # Steps before optimizer update
epochs: int = 3                         # Training epochs
weight_decay: float = 0.01              # AdamW weight decay
max_grad_norm: float = 1.0              # Gradient clipping
```

### Data (`DataConfig`)

```python
current_datasets: [SYNTHETIC, FINEWEB]  # Datasets to use
dataset_probabilities: [0.7, 0.3]       # Sampling weights
max_samples: int = 5000000              # Samples per dataset
max_length: int = 1024                  # Sequence length
streaming: bool = True                  # Stream from HuggingFace
```

---

## Vast.ai Deployment

### 1. Rent GPUs

Go to [vast.ai](https://vast.ai) and rent a machine with:
- 4-8x A100 (40GB or 80GB)
- Ubuntu image with CUDA
- Adequate disk space (~50GB)

### 2. Connect & Setup

```bash
# SSH into the machine
ssh -p <port> root@<ip-address>

# Clone repository
git clone <your-repo-url>
cd BasicGPT

# Install dependencies
pip install -r requirements.txt
```

### 3. Sync Files (from local machine)

```bash
# Sync your local changes to vast.ai
rsync -avz -e "ssh -p <port>" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints' \
    --exclude='.git' \
    --exclude='.venv' \
    ./ root@<ip>:/root/BasicGPT/
```

### 4. Run Training

```bash
# On vast.ai machine
cd BasicGPT

# Multi-GPU training
accelerate launch --multi_gpu train.py
```

### 5. Download Checkpoints

```bash
# From local machine
rsync -avz -e "ssh -p <port>" \
    root@<ip>:/root/BasicGPT/checkpoints/ \
    ./checkpoints/
```

---

## Effective Batch Size

The effective batch size scales with GPUs:

```
Effective = batch_size × gradient_accumulation × num_gpus

Single GPU:  32 × 4 × 1 = 128
4x A100:     32 × 4 × 4 = 512
8x A100:     32 × 4 × 8 = 1024
```

For A100 40GB, you may need to reduce `batch_size` to 16-24.

---

## Checkpoints

Checkpoints are saved to `./checkpoints/` with this structure:

```
checkpoints/
├── data-5m-batch-10000-11-26-2025-14-30/
│   └── checkpoint.pt
├── data-5m-batch-20000-11-26-2025-15-00/
│   └── checkpoint.pt
└── checkpoint_best.pt    # Best validation loss
```

Each checkpoint contains:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state
- `scheduler_state_dict` - LR scheduler state
- `gpt_config` - Model architecture
- `training_config` - Training hyperparameters
- `train_loss`, `val_loss` - Loss values
- `epoch`, `total_batches` - Training progress

---

## Datasets

Currently supported datasets:

| Dataset | Description | Default Weight |
|---------|-------------|----------------|
| `SYNTHETIC` | PleIAs/SYNTH - High-quality Q&A pairs | 70% |
| `FINEWEB` | HuggingFaceFW/fineweb - Web text | 30% |
| `C4` | allenai/c4 - Colossal Clean Crawled Corpus | - |

Datasets are streamed from HuggingFace Hub (no local download required).

### Changing Datasets

Edit `config.py`:

```python
@dataclass
class DataConfig:
    current_datasets: list[DatasetName] = field(default_factory=lambda: [
        DatasetName.SYNTHETIC,
        DatasetName.FINEWEB,
        DatasetName.C4,  # Add more
    ])
    dataset_probabilities: Optional[list[float]] = field(
        default_factory=lambda: [0.5, 0.3, 0.2]  # Must match dataset count
    )
```

---

## Helper Scripts

Located in `scripts/`:

| Script | Description |
|--------|-------------|
| `calc_params.py` | Calculate model parameters and memory requirements |
| `calc_lr_config.py` | Calculate learning rate schedule parameters |
| `verify_lr.py` | Visualize learning rate schedule |
| `clean_data.py` | Data cleaning utilities |
| `vast.sh` | Vast.ai SSH/rsync commands |

---

## Requirements

```
torch>=2.0
tiktoken
datasets
accelerate
```

Install with:
```bash
pip install -r requirements.txt
```

---

## License

MIT

