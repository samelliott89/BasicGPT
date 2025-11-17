"""
Data preparation script for loading and processing the SYNTH dataset.

This script loads the SYNTH dataset from Hugging Face, tokenizes the text,
and creates PyTorch datasets and data loaders for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizer import Tokenizer
from typing import Optional

class SYNTHDataset(Dataset):
    """
    A PyTorch Dataset class for the SYNTH dataset.
    
    This class handles:
    - Loading text samples from the SYNTH dataset
    - Tokenizing the text using our Tokenizer
    - Creating sequences of the correct length for training
    - Preparing input and target sequences for language modeling
    
    In language modeling, we predict the next token given previous tokens.
    So if we have tokens [1, 2, 3, 4, 5], the input is [1, 2, 3, 4]
    and the target is [2, 3, 4, 5] (shifted by one position).
    """
    
    def __init__(
        self,
        dataset,
        tokenizer: Tokenizer,
        max_length: int = 1024,
        text_field: str = "synthetic_answer"
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset: The Hugging Face dataset (already loaded)
            tokenizer: Our Tokenizer instance for encoding text
            max_length: Maximum sequence length (context window size)
            text_field: Which field from the dataset to use as text
                       Options: "synthetic_answer", "query", "synthetic_reasoning"
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        
        # Pre-tokenize all samples for faster training
        # This converts all text to token IDs upfront
        print(f"Tokenizing {len(dataset)} samples...")
        self.tokenized_samples = []
        
        for i, sample in enumerate(dataset):
            # Get the text from the specified field
            text = sample.get(text_field, "")
            
            # Combine multiple fields if needed for richer context
            # You can modify this to include query + reasoning + answer
            if text_field == "synthetic_answer":
                # Optionally combine with query for better context
                query = sample.get("query", "")
                if query:
                    # Combine query and answer with a separator
                    text = f"{query}\n\n{text}"
            
            # Skip empty samples
            if not text or len(text.strip()) == 0:
                continue
            
            # Tokenize the text
            tokens = self.tokenizer.encode(text)
            
            # Only keep samples that have at least some tokens
            if len(tokens) > 0:
                self.tokenized_samples.append(tokens)
            
            # Progress indicator
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")
        
        print(f"✓ Tokenized {len(self.tokenized_samples)} valid samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.tokenized_samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            A tuple of (input_ids, target_ids) where:
            - input_ids: Token IDs for the input sequence [0:max_length-1]
            - target_ids: Token IDs for the target sequence [1:max_length]
            
            The target is shifted by one position because we're predicting
            the next token at each position.
        """
        # Get the tokenized sample
        tokens = self.tokenized_samples[idx]
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            # If too long, take the first max_length tokens
            tokens = tokens[:self.max_length]
        else:
            # If too short, pad with a special token (0 is often used for padding)
            # Note: tiktoken doesn't have a special padding token, so we use 0
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # Convert to PyTorch tensor
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create input and target sequences
        # Input: tokens from position 0 to max_length-1
        # Target: tokens from position 1 to max_length (shifted by 1)
        input_ids = token_tensor[:-1]  # [0, 1, 2, ..., max_length-2]
        target_ids = token_tensor[1:]  # [1, 2, 3, ..., max_length-1]
        
        return input_ids, target_ids


def load_synth_dataset(
    tokenizer: Tokenizer,
    max_length: int = 1024,
    split: str = "train",
    streaming: bool = False,
    max_samples: Optional[int] = None,
    text_field: str = "synthetic_answer"
) -> SYNTHDataset:
    """
    Load the SYNTH dataset from Hugging Face and prepare it for training.
    
    Args:
        tokenizer: The Tokenizer instance to use
        max_length: Maximum sequence length
        split: Dataset split to load ("train", "validation", etc.)
        streaming: If True, stream the dataset (for very large datasets)
        max_samples: Maximum number of samples to load (None = all)
        text_field: Which field to use as text source
        
    Returns:
        A SYNTHDataset instance ready for training
    """
    print(f"Loading SYNTH dataset from Hugging Face...")
    print(f"  Split: {split}")
    print(f"  Streaming: {streaming}")
    print(f"  Max samples: {max_samples if max_samples else 'all'}")
    
    # Load the dataset from Hugging Face
    # The SYNTH dataset is quite large (~68M samples), so we might want to
    # use streaming=True for initial experiments
    dataset = load_dataset(
        "PleIAs/SYNTH",
        split=split,
        streaming=streaming
    )
    
    # If max_samples is specified and not streaming, take a subset
    if max_samples and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    elif max_samples and streaming:
        # For streaming, we'll take first max_samples
        dataset = dataset.take(max_samples)
    
    # Create our custom dataset
    synth_dataset = SYNTHDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        text_field=text_field
    )
    
    return synth_dataset


def create_data_loaders(
    train_dataset: SYNTHDataset,
    val_dataset: Optional[SYNTHDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    DataLoaders handle:
    - Batching multiple samples together
    - Shuffling data for training
    - Parallel data loading for efficiency
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Number of samples per batch
        num_workers: Number of parallel workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training data loader
    # shuffle=True randomizes the order of samples each epoch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    # Create validation data loader (if provided)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    print(f"✓ Tokenizer initialized with vocab_size={tokenizer.vocab_size}")
    print()
    
    # Load a small subset for testing (use streaming for large datasets)
    print("Loading SYNTH dataset (small subset for testing)...")
    train_dataset = load_synth_dataset(
        tokenizer=tokenizer,
        max_length=512,  # Smaller for testing
        split="train",
        streaming=True,  # Use streaming for large datasets
        max_samples=1000,  # Just 1000 samples for testing
        text_field="synthetic_answer"
    )
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, _ = create_data_loaders(
        train_dataset=train_dataset,
        batch_size=4,  # Small batch for testing
        num_workers=0  # Set to 0 for debugging, increase for production
    )
    print(f"✓ Created train loader with {len(train_loader)} batches")
    print()
    
    # Test a batch
    print("Testing a batch...")
    input_ids, target_ids = next(iter(train_loader))
    print(f"  Input shape: {input_ids.shape}")  # Should be [batch_size, max_length-1]
    print(f"  Target shape: {target_ids.shape}")  # Should be [batch_size, max_length-1]
    print()
    
    # Decode a sample to verify
    print("Decoding a sample to verify tokenization...")
    sample_input = input_ids[0].tolist()
    sample_text = tokenizer.decode(sample_input[:50])  # First 50 tokens
    print(f"  Sample text (first 50 tokens): {sample_text[:200]}...")
    print()
    
    print("✓ Data preparation complete!")

