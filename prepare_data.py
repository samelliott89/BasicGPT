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
from config import GPTConfig, DataConfig, TrainingConfig

gpt_config = GPTConfig()
data_config = DataConfig()
training_config = TrainingConfig()

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
    max_samples: int = data_config.max_samples,
    val_split_percentage: float = 0.1,  # Use 10% for validation
    use_val_split: bool = False  # If True, skip first (1-val_split_percentage) of data
) -> SYNTHDataset:
    """
    Load the SYNTH dataset from Hugging Face and prepare it for training.
    
    This function includes retry logic to handle network timeouts and connection issues.
    Since the PleIAs/SYNTH dataset doesn't have a validation split, this function can
    split the train data into train/val portions.
    
    Args:
        tokenizer: The Tokenizer instance to use
        max_length: Maximum sequence length
        split: Dataset split to load ("train", "validation", etc.)
               Note: PleIAs/SYNTH only has "train", so always use "train"
        streaming: If True, stream the dataset (recommended for large datasets)
                  Streaming doesn't download the entire dataset upfront, which
                  helps avoid timeout issues.
        max_samples: Maximum number of samples to load (None = all)
        text_field: Which field to use as text source
        include_reasoning: If True, include reasoning steps in training data
        num_retries: Number of times to retry on failure
        timeout: Timeout in seconds for download operations
        val_split_percentage: Percentage of data to use for validation (0.0 to 1.0)
        use_val_split: If True, return the validation portion (last val_split_percentage)
                      If False, return the train portion (first 1-val_split_percentage)
        
    Returns:
        A SYNTHDataset instance ready for training
        
    Raises:
        FileNotFoundError: If dataset cannot be loaded after retries
    """
    print(f"Loading SYNTH dataset from Hugging Face...")
    print(f"  Split: {split} (loading from 'train' split)")
    if use_val_split:
        print(f"  Using VALIDATION portion (last {val_split_percentage*100:.0f}% of data)")
    else:
        print(f"  Using TRAINING portion (first {(1-val_split_percentage)*100:.0f}% of data)")
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
            # NOTE: PleIAs/SYNTH only has "train" split, no validation split
            print("Connecting to Hugging Face Hub...")
            dataset = load_dataset(
                "PleIAs/SYNTH",
                split="train",  # Always use "train" - we'll split it ourselves
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
    
    # Split the dataset into train/validation portions
    # For streaming datasets, we need to use skip() and take()
    if max_samples:
        if use_val_split:
            # For validation: skip training portion, take validation portion
            train_samples = int(max_samples * (1 - val_split_percentage))
            val_samples = max_samples - train_samples
            dataset = dataset.skip(train_samples).take(val_samples)
        else:
            # For training: take training portion only
            train_samples = int(max_samples * (1 - val_split_percentage))
            dataset = dataset.take(train_samples)
    else:
        # No max_samples specified - need to handle differently for streaming
        if streaming:
            # For streaming without max_samples, we can't easily split
            # We'll need to rely on the caller to specify max_samples
            # For now, print a warning
            if use_val_split:
                print("⚠️  WARNING: Cannot create validation split from streaming dataset without max_samples")
                print("   Please specify max_samples to enable train/val splitting")
        else:
            # For non-streaming, we can use the full dataset and split it
            total_samples = len(dataset)
            if use_val_split:
                # Validation: take last val_split_percentage
                train_samples = int(total_samples * (1 - val_split_percentage))
                dataset = dataset.select(range(train_samples, total_samples))
            else:
                # Training: take first (1 - val_split_percentage)
                train_samples = int(total_samples * (1 - val_split_percentage))
                dataset = dataset.select(range(train_samples))


    # Shuffle the dataset if it's for training (only if streaming/iterable)
    # For streaming datasets, we can use buffer-based shuffling
    # This shuffles within a buffer window, providing good randomization
    if isinstance(dataset, HfIterableDataset) or streaming:
        # Only shuffle training data, not validation
        if not use_val_split or (use_val_split and streaming):
            # For iterable datasets, use buffer-based shuffle
            # buffer_size determines how many samples to load before shuffling
            # Larger buffer = better shuffle quality but more memory usage
            dataset = dataset.shuffle(buffer_size=10000, seed=42)
            print("  Applied buffer-based shuffling (buffer_size=10000)")
    
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



def load_training_data(
    dataset_class,
    dataset_name: str,
    tokenizer: Tokenizer,
    name: str = None,
    max_length: int = None,
    text_field: str = "text",
    split: str = "train",
    streaming: bool = None,
    num_retries: int = None,
    timeout: int = None,
    max_samples: int = None,
    val_split_percentage: float = 0.1,
    has_val_split: bool = False,
    **dataset_kwargs
):
    """
    Generic helper to load training data for any dataset.
    
    This function handles two scenarios:
    1. Dataset has a native validation split: loads from split="train"
    2. Dataset needs manual splitting: loads from split="train" with use_val_split=False (first 90%)
    
    Args:
        dataset_class: The dataset class to instantiate (e.g., FineWebDataset, C4Dataset)
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb")
        tokenizer: Tokenizer instance for encoding text
        name: Optional dataset configuration name (e.g., "sample-10BT", "en")
        max_length: Maximum sequence length (default from DataConfig)
        text_field: Which field contains the text (default: "text")
        split: Dataset split to load (default: "train")
        streaming: Whether to stream the dataset (default from DataConfig)
        num_retries: Number of retry attempts (default from DataConfig)
        timeout: Timeout for downloads (default from DataConfig)
        max_samples: Maximum number of samples (default from DataConfig)
        val_split_percentage: Percentage for validation split (default: 0.1 = 10%)
        has_val_split: If True, dataset has native validation split
                       If False, will split train data 90/10 (use_val_split=False for training)
        **dataset_kwargs: Additional dataset-specific parameters (e.g., include_reasoning, filter_english_only)
        
    Returns:
        Dataset instance ready for training
        
    Example:
        # For FineWeb (no native val split, needs manual splitting)
        train_ds = load_training_data(
            dataset_class=FineWebDataset,
            dataset_name="HuggingFaceFW/fineweb",
            name="sample-10BT",
            tokenizer=tokenizer,
            has_val_split=False  # Will use use_val_split=False to get first 90%
        )
        
        # For a dataset with native validation split
        train_ds = load_training_data(
            dataset_class=SomeDataset,
            dataset_name="some/dataset",
            tokenizer=tokenizer,
            has_val_split=True  # Will just load from split="train"
        )
    """
    # Get defaults from DataConfig
    data_config_instance = DataConfig()
    if max_length is None:
        max_length = data_config_instance.max_length
    
    # Load the raw dataset
    from datasets.datasetprep import DatasetPrep
    
    # For training data, we always use use_val_split=False
    # - If has_val_split=True: loads full "train" split (dataset has separate validation)
    # - If has_val_split=False: loads first 90% of "train" split
    raw_dataset = DatasetPrep.load_dataset(
        dataset_name=dataset_name,
        name=name,
        split=split,
        streaming=streaming,
        num_retries=num_retries,
        timeout=timeout,
        max_samples=max_samples,
        use_val_split=False,  # Always False for training
        val_split_percentage=val_split_percentage,
    )
    
    # Wrap with the dataset-specific class
    return dataset_class(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        text_field=text_field,
        **dataset_kwargs
    )



def load_validation_data(
    dataset_class,
    dataset_name: str,
    tokenizer: Tokenizer,
    name: str = None,
    max_length: int = None,
    text_field: str = "text",
    split: str = "validation",
    streaming: bool = None,
    num_retries: int = None,
    timeout: int = None,
    max_samples: int = None,
    val_split_percentage: float = 0.1,
    has_val_split: bool = False,
    **dataset_kwargs
):
    """
    Generic helper to load validation data for any dataset.
    
    This function handles two scenarios:
    1. Dataset has a native validation split: loads from split="validation"
    2. Dataset needs manual splitting: loads from split="train" with use_val_split=True
    
    Args:
        dataset_class: The dataset class to instantiate (e.g., FineWebDataset, C4Dataset)
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb")
        tokenizer: Tokenizer instance for encoding text
        name: Optional dataset configuration name (e.g., "sample-10BT", "en")
        max_length: Maximum sequence length (default from DataConfig)
        text_field: Which field contains the text (default: "text")
        split: Dataset split to load (default: "validation", or "train" if no native split)
        streaming: Whether to stream the dataset (default from DataConfig)
        num_retries: Number of retry attempts (default from DataConfig)
        timeout: Timeout for downloads (default from DataConfig)
        max_samples: Maximum number of samples (default from DataConfig)
        val_split_percentage: Percentage for validation split (default: 0.1 = 10%)
        has_val_split: If True, dataset has native validation split, use split="validation"
                       If False, use use_val_split=True to get last 10% of train data
        **dataset_kwargs: Additional dataset-specific parameters (e.g., include_reasoning, filter_english_only)
        
    Returns:
        Dataset instance ready for validation
        
    Example:
        # For FineWeb (no native val split, needs manual splitting)
        val_ds = load_validation_data(
            dataset_class=FineWebDataset,
            dataset_name="HuggingFaceFW/fineweb",
            name="sample-10BT",
            tokenizer=tokenizer,
            has_val_split=False  # Will use use_val_split=True on train split
        )
        
        # For a dataset with native validation split
        val_ds = load_validation_data(
            dataset_class=SomeDataset,
            dataset_name="some/dataset",
            tokenizer=tokenizer,
            has_val_split=True  # Will use split="validation"
        )
    """
    # Get defaults from DataConfig
    data_config_instance = DataConfig()
    if max_length is None:
        max_length = data_config_instance.max_length
    
    # Load the raw dataset
    from datasets.datasetprep import DatasetPrep
    
    if has_val_split:
        # Dataset has native validation split, load from "validation" split
        raw_dataset = DatasetPrep.load_dataset(
            dataset_name=dataset_name,
            name=name,
            split=split,
            streaming=streaming,
            num_retries=num_retries,
            timeout=timeout,
            max_samples=max_samples,
            use_val_split=False,  # Not needed, using native split
            val_split_percentage=val_split_percentage,
        )
    else:
        # Dataset needs manual splitting, get last 10% (validation portion)
        raw_dataset = DatasetPrep.load_dataset(
            dataset_name=dataset_name,
            name=name,
            split="train",  # Load from train split, then take last 10%
            streaming=streaming,
            num_retries=num_retries,
            timeout=timeout,
            max_samples=max_samples,
            use_val_split=True,  # Get validation portion (last 10%)
            val_split_percentage=val_split_percentage,
        )
    
    # Wrap with the dataset-specific class
    return dataset_class(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        text_field=text_field,
        **dataset_kwargs
    )
    

def create_data_loaders(
    train_dataset,
    val_dataset = None,
    batch_size: int = data_config.batch_size,  # Default batch size (should match TrainingConfig default)
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
    
    # Note: Shuffling is already handled in load_synth_dataset for streaming datasets
    # For non-streaming datasets, we use DataLoader's shuffle parameter below
    
    # Create training data loader
    # Note: IterableDataset doesn't support shuffle, so we skip it for streaming mode
    # Also, prefetch_factor and persistent_workers only work with num_workers > 0
    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": not is_iterable,
        "num_workers": num_workers if not is_iterable else 0,
        "pin_memory": True,
    }
    
    # Only add these options if using multiple workers
    if train_loader_kwargs["num_workers"] > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        train_loader_kwargs["persistent_workers"] = True
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    # Create validation data loader (if provided)
    val_loader = None
    if val_dataset is not None:
        is_val_iterable = isinstance(val_dataset, IterableDataset)
        val_loader_kwargs = {
            "batch_size": batch_size * 2,  # Can be larger (no gradients stored)
            "shuffle": False,  # No shuffling needed for validation
            "num_workers": num_workers if not is_val_iterable else 0,
            "pin_memory": True,
        }
        
        # Only add these options if using multiple workers
        if val_loader_kwargs["num_workers"] > 0:
            val_loader_kwargs["prefetch_factor"] = 2
            val_loader_kwargs["persistent_workers"] = True
        
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
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

