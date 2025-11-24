# Dataset Architecture - Usage Examples

This document shows how to use the refactored dataset architecture.

## Architecture Overview

The dataset architecture uses inheritance to share common functionality:

- **`DatasetPrep`** (Abstract Base Class): Provides common methods for tokenization, iteration, and dataset loading
- **`FineWebDataset`**, **`SYNTHIterableDataset`** (Concrete Classes): Implement dataset-specific preprocessing

### Key Methods

Each dataset class has:
- `_pre_process_sample(sample)`: Dataset-specific text extraction (implemented by subclass)
- `_tokenize_sample(text)`: Common tokenization logic (inherited from `DatasetPrep`)
- `__iter__()`: Common iteration logic (inherited from `DatasetPrep`)
- `load_dataset(...)`: Static method for loading from HuggingFace (inherited from `DatasetPrep`)
- `from_huggingface(...)`: Class method for convenient instantiation (implemented by subclass)

## Usage Examples

### Example 1: Load FineWeb Dataset

```python
from tokenizer import Tokenizer
from datasets.finewebdataset import FineWebDataset

# Initialize tokenizer
tokenizer = Tokenizer()

# Load FineWeb dataset using the class method
dataset = FineWebDataset.from_huggingface(
    dataset_name="HuggingFaceFW/fineweb",
    tokenizer=tokenizer,
    max_length=1024,
    text_field="text",
    streaming=True,
    max_samples=10000,
    use_val_split=False  # Training data
)

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    input_ids, target_ids = batch
    # Train your model...
```

### Example 2: Load SYNTH Dataset

```python
from tokenizer import Tokenizer
from datasets.synthdataset import SYNTHIterableDataset

# Initialize tokenizer
tokenizer = Tokenizer()

# Load SYNTH dataset with reasoning included
train_dataset = SYNTHIterableDataset.from_huggingface(
    dataset_name="PleIAs/SYNTH",
    tokenizer=tokenizer,
    max_length=1024,
    text_field="synthetic_answer",
    include_reasoning=True,  # Include reasoning steps
    filter_english_only=True,  # Only English samples
    streaming=True,
    max_samples=50000,
    use_val_split=False,  # Training portion
    val_split_percentage=0.1
)

# Load validation split
val_dataset = SYNTHIterableDataset.from_huggingface(
    dataset_name="PleIAs/SYNTH",
    tokenizer=tokenizer,
    max_length=1024,
    text_field="synthetic_answer",
    include_reasoning=True,
    filter_english_only=True,
    streaming=True,
    max_samples=50000,
    use_val_split=True,  # Validation portion
    val_split_percentage=0.1
)
```

### Example 3: Create a Custom Dataset

To create a new dataset, inherit from `DatasetPrep` and implement `_pre_process_sample`:

```python
from typing import Optional
from datasets.datasetprep import DatasetPrep

class MyCustomDataset(DatasetPrep):
    """Custom dataset with specific preprocessing logic."""
    
    def __init__(self, dataset, tokenizer, max_length, **kwargs):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs
        )
    
    def _pre_process_sample(self, sample: dict) -> Optional[str]:
        """
        Implement your custom preprocessing logic here.
        
        Returns:
            Preprocessed text string, or None to skip the sample
        """
        # Example: Extract and combine multiple fields
        title = sample.get('title', '')
        content = sample.get('content', '')
        
        if not content:
            return None  # Skip samples without content
        
        # Combine fields
        text = f"{title}\n\n{content}" if title else content
        
        return text
    
    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        tokenizer,
        max_length: int = 1024,
        **kwargs
    ):
        """Load dataset and create instance."""
        # Load raw dataset
        dataset = DatasetPrep.load_dataset(
            dataset_name=dataset_name,
            **kwargs
        )
        
        # Create and return instance
        return cls(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length
        )
```

## Benefits of This Architecture

1. **Code Reuse**: Tokenization and iteration logic is written once in `DatasetPrep`
2. **Flexibility**: Each dataset can implement its own preprocessing in `_pre_process_sample`
3. **Consistency**: All datasets follow the same pattern and interface
4. **Easy Extension**: Adding a new dataset only requires implementing `_pre_process_sample`
5. **Type Safety**: Abstract base class ensures all datasets implement required methods
