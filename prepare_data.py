"""
Data preparation script for loading and processing the SYNTH dataset.

This script loads the SYNTH dataset from Hugging Face, tokenizes the text,
and creates PyTorch datasets and data loaders for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset as HfIterableDataset
from tokenizer import Tokenizer
from typing import Optional
import time
import os
from config import GPTConfig, DataConfig

gpt_config = GPTConfig()
data_config = DataConfig()

class SYNTHDataset(Dataset):
#     """
#     A PyTorch Dataset class for the SYNTH dataset (non-streaming mode).
    
#     This class handles:
#     - Loading text samples from the SYNTH dataset
#     - Tokenizing the text using our Tokenizer
#     - Creating sequences of the correct length for training
#     - Preparing input and target sequences for language modeling
    
#     In language modeling, we predict the next token given previous tokens.
#     So if we have tokens [1, 2, 3, 4, 5], the input is [1, 2, 3, 4]
#     and the target is [2, 3, 4, 5] (shifted by one position).
#     """
    
    def __init__(
        self,
        dataset,
        tokenizer: Tokenizer,
        max_length: int = data_config.max_length,
        text_field: str = "synthetic_answer"
    ):
#         """
#         Initialize the dataset.
        
#         Args:
#             dataset: The Hugging Face dataset (non-streaming, regular dataset)
#             tokenizer: Our Tokenizer instance for encoding text
#             max_length: Maximum sequence length (context window size)
#             text_field: Which field from the dataset to use as text
#                        Options: "synthetic_answer", "query", "synthetic_reasoning"
#         """
#         self.dataset = dataset
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.text_field = text_field
        
#         # Pre-tokenize all samples for faster training
#         # This converts all text to token IDs upfront
#         print(f"Tokenizing {len(dataset)} samples...")
#         self.tokenized_samples = []
        
#         for i, sample in enumerate(dataset):
#             # Get the text from the specified field
#             text = sample.get(text_field, "")
            
#             # Combine multiple fields if needed for richer context
#             # You can modify this to include query + reasoning + answer
#             if text_field == "synthetic_answer":
#                 # Optionally combine with query for better context
#                 query = sample.get("query", "")
#                 if query:
#                     # Combine query and answer with a separator
#                     text = f"{query}\n\n{text}"
            
#             # Skip empty samples
#             if not text or len(text.strip()) == 0:
#                 continue
            
#             # Tokenize the text
#             tokens = self.tokenizer.encode(text)
            
#             # Only keep samples that have at least some tokens
#             if len(tokens) > 0:
#                 self.tokenized_samples.append(tokens)
            
#             # Progress indicator
#             if (i + 1) % 10000 == 0:
#                 print(f"  Processed {i + 1}/{len(dataset)} samples...")
        
#         print(f"✓ Tokenized {len(self.tokenized_samples)} valid samples")
    
#     def __len__(self) -> int:
#         """Return the number of samples in the dataset."""
#         return len(self.tokenized_samples)
    
#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Get a single training sample.
        
#         Returns:
#             A tuple of (input_ids, target_ids) where:
#             - input_ids: Token IDs for the input sequence [0:max_length-1]
#             - target_ids: Token IDs for the target sequence [1:max_length]
            
#             The target is shifted by one position because we're predicting
#             the next token at each position.
#         """
#         # Get the tokenized sample
#         tokens = self.tokenized_samples[idx]
        
#         # Truncate or pad to max_length
#         if len(tokens) > self.max_length:
#             # If too long, take the first max_length tokens
#             tokens = tokens[:self.max_length]
#         else:
#             # If too short, pad with a special token (0 is often used for padding)
#             # Note: tiktoken doesn't have a special padding token, so we use 0
#             tokens = tokens + [0] * (self.max_length - len(tokens))
        
#         # Convert to PyTorch tensor
#         token_tensor = torch.tensor(tokens, dtype=torch.long)
        
#         # Create input and target sequences
#         # Input: tokens from position 0 to max_length-1
#         # Target: tokens from position 1 to max_length (shifted by 1)
#         input_ids = token_tensor[:-1]  # [0, 1, 2, ..., max_length-2]
#         target_ids = token_tensor[1:]  # [1, 2, 3, ..., max_length-1]
        
#         return input_ids, target_ids
        pass


class SYNTHIterableDataset(IterableDataset):
    """
    A PyTorch IterableDataset class for the SYNTH dataset (streaming mode).
    
    This class handles streaming datasets that don't support len() or indexing.
    It tokenizes samples on-the-fly as they're requested.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer: Tokenizer,
        max_length: int = data_config.max_length,
        text_field: str = "synthetic_answer",
        include_reasoning: bool = False,
        filter_english_only: bool = True
    ):
        """
        Initialize the iterable dataset.
        
        Args:
            dataset: The Hugging Face IterableDataset (streaming mode)
            tokenizer: Our Tokenizer instance for encoding text
            max_length: Maximum sequence length (context window size)
            text_field: Which field from the dataset to use as text
            include_reasoning: If True, include reasoning steps in training data
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.include_reasoning = include_reasoning
        self.filter_english_only = filter_english_only
    
    def _process_sample(self, sample: dict) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a single sample: extract text, tokenize, and format.
        
        Returns:
            A tuple of (input_ids, target_ids) or None if sample is invalid
        """
        # Get the text from the specified field
        text = sample.get(self.text_field, "")
        
        # Combine multiple fields if needed for richer context
        if self.text_field == "synthetic_answer":
            query = sample.get("query", "")
            reasoning = sample.get("synthetic_reasoning", "") if self.include_reasoning else ""
            answer = text
            
            # Build text based on available fields and include_reasoning flag
            if query and reasoning and answer:
                text = f"{query}\n\n{reasoning}\n\n{answer}"
            elif query and answer:
                text = f"{query}\n\n{answer}"
            elif answer:
                text = answer
            elif query:
                text = query
        
        # Skip empty samples
        if not text or len(text.strip()) == 0:
            return None
        
        # Filter for English-only if enabled (using dataset's language field)
        if self.filter_english_only:
            sample_language = sample.get('language')
            if sample_language:
                # Dataset uses ISO language codes: 'en', 'de', 'fr', etc.
                # Only keep English samples
                if str(sample_language).lower() != 'en':
                    return None  # Skip non-English samples
            else:
                # If no language field, warn but continue (dataset might not have it)
                # In practice, SYNTH dataset should have this field
                pass
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        # Skip if no tokens
        if len(tokens) == 0:
            return None
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # Clamp token IDs to valid vocabulary range [0, vocab_size-1]
        # This prevents index out of bounds errors in embedding layers
        vocab_size = self.tokenizer.vocab_size
        tokens = [min(max(token, 0), vocab_size - 1) for token in tokens]
        
        # Convert to PyTorch tensor
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create input and target sequences (shifted by 1)
        input_ids = token_tensor[:-1]
        target_ids = token_tensor[1:]
        
        return input_ids, target_ids
    
    def __iter__(self):
        """
        Iterate over the dataset, yielding processed samples.
        
        This is called by PyTorch's DataLoader for IterableDataset.
        """
        for sample in self.dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                yield processed


def load_synth_dataset(
    tokenizer: Tokenizer,
    max_length: int = data_config.max_length,
    split: str = "train",
    streaming: bool = data_config.streaming,  # Default to streaming to avoid large downloads
    text_field: str = "synthetic_answer",
    include_reasoning: bool = data_config.include_reasoning,
    filter_english_only: bool = data_config.filter_english_only,
    num_retries: int = data_config.num_retries,
    timeout: int = data_config.timeout,
    max_samples: int = data_config.max_samples
) -> SYNTHDataset:
    """
    Load the SYNTH dataset from Hugging Face and prepare it for training.
    
    This function includes retry logic to handle network timeouts and connection issues.
    
    Args:
        tokenizer: The Tokenizer instance to use
        max_length: Maximum sequence length
        split: Dataset split to load ("train", "validation", etc.)
        streaming: If True, stream the dataset (recommended for large datasets)
                  Streaming doesn't download the entire dataset upfront, which
                  helps avoid timeout issues.
        max_samples: Maximum number of samples to load (None = all)
        text_field: Which field to use as text source
        include_reasoning: If True, include reasoning steps in training data
        num_retries: Number of times to retry on failure
        timeout: Timeout in seconds for download operations
        
    Returns:
        A SYNTHDataset instance ready for training
        
    Raises:
        FileNotFoundError: If dataset cannot be loaded after retries
    """
    print(f"Loading SYNTH dataset from Hugging Face...")
    print(f"  Split: {split}")
    print(f"  Streaming: {streaming} (recommended for large datasets)")
    print(f"  Max samples: {max_samples if max_samples else 'all'}")
    print(f"  Include reasoning: {include_reasoning}")
    print(f"  Filter English only: {filter_english_only}")
    print(f"  Timeout: {timeout} seconds")
    print()
    
    # Set environment variable for longer timeout
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
    
    # Retry logic with exponential backoff
    last_error = None
    for attempt in range(num_retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                print(f"Retry attempt {attempt + 1}/{num_retries} after {wait_time} seconds...")
                time.sleep(wait_time)
            
            # Load the dataset from Hugging Face
            # The SYNTH dataset is quite large (~68M samples), so streaming=True
            # is recommended as it doesn't require downloading everything upfront
            print("Connecting to Hugging Face Hub...")
            dataset = load_dataset(
                "PleIAs/SYNTH",
                split=split,
                streaming=streaming,
                download_config={
                    "timeout": timeout,
                    "num_proc": 1,  # Reduce parallel downloads to avoid timeouts
                } if not streaming else None
            )
            
            print("✓ Successfully connected to dataset")
            break
            
        except (FileNotFoundError, ConnectionError, TimeoutError) as e:
            last_error = e
            error_msg = str(e)
            
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"✗ Connection timeout (attempt {attempt + 1}/{num_retries})")
                if attempt < num_retries - 1:
                    print("  This might be due to:")
                    print("  - Slow internet connection")
                    print("  - Network firewall/proxy issues")
                    print("  - Hugging Face Hub being temporarily unavailable")
                    print("  - Dataset being very large")
                    print()
                    if not streaming:
                        print("  💡 Tip: Try using --streaming flag to avoid downloading the entire dataset")
                        print()
            elif "connection" in error_msg.lower():
                print(f"✗ Connection error (attempt {attempt + 1}/{num_retries})")
                print("  Please check your internet connection")
                print()
            else:
                print(f"✗ Error loading dataset (attempt {attempt + 1}/{num_retries}): {error_msg}")
                print()
            
            if attempt == num_retries - 1:
                # Last attempt failed
                print("=" * 60)
                print("Failed to load dataset after all retry attempts.")
                print()
                print("Troubleshooting tips:")
                print("1. Check your internet connection")
                print("2. Try using --streaming flag (recommended for large datasets)")
                print("3. Increase timeout with: export HF_HUB_DOWNLOAD_TIMEOUT=600")
                print("4. Try again later (Hugging Face Hub might be temporarily unavailable)")
                print("5. Check if you can access https://huggingface.co/datasets/PleIAs/SYNTH")
                print("=" * 60)
                raise FileNotFoundError(
                    f"Could not load SYNTH dataset after {num_retries} attempts. "
                    f"Last error: {error_msg}"
                ) from last_error
    
    # If max_samples is specified and not streaming, take a subset
    if max_samples and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    elif max_samples and streaming:
        # For streaming, we'll take first max_samples
        dataset = dataset.take(max_samples)
    
    # Create our custom dataset
    # Check if it's a streaming dataset (IterableDataset)
    if isinstance(dataset, HfIterableDataset) or streaming:
        print("Using IterableDataset for streaming mode...")
        synth_dataset = SYNTHIterableDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            text_field=text_field,
            include_reasoning=include_reasoning,
            filter_english_only=filter_english_only
        )
    else:
        print("Using regular Dataset (pre-tokenizing all samples)...")
        synth_dataset = SYNTHDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            text_field=text_field
        )
    
    return synth_dataset
    

def create_data_loaders(
    train_dataset,
    val_dataset = None,
    batch_size: int = 16,  # Default batch size (should match TrainingConfig default)
    num_workers: int = data_config.num_dataset_workers,

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
    # Check if dataset is an IterableDataset (streaming mode)
    is_iterable = isinstance(train_dataset, IterableDataset)

    # Shuffle training dataset
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=42)
    
    # Create training data loader
    # Note: IterableDataset doesn't support shuffle, so we skip it for streaming mode
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not is_iterable,
        num_workers=num_workers ## if not is_iterable else 0,  # IterableDataset needs num_workers=0
        pin_memory=True,    
        prefetch_factor=2,
        persistent_workers=True, 
    )
    
    # Create validation data loader (if provided)
    val_loader = None
    if val_dataset is not None:
        is_val_iterable = isinstance(val_dataset, IterableDataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,          # ← Can be larger (no gradients stored)
            shuffle=False,          # ← No shuffling needed
            num_workers=4,          # ← Same
            pin_memory=True,        # ← Same
            prefetch_factor=2,      # ← Same
            persistent_workers=True # ← Same
        )
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     shuffle=False,  # Don't shuffle validation data
        #     num_workers=num_workers if not is_val_iterable else 0,
        #     pin_memory=True
        # )
    
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

