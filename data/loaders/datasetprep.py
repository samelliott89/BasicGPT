from abc import ABC, abstractmethod
from typing import Optional
import torch
import os
import time
from torch.utils.data import IterableDataset
from datasets import load_dataset
from datasets import IterableDataset as HfIterableDataset
import sys
from pathlib import Path
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DataConfig as data_config
from enums import DatasetName, DatasetLang, DatasetSplit

class DatasetPrep(IterableDataset, ABC):
    """
    Abstract base class for dataset preparation.
    
    Provides common functionality for:
    - Tokenization (_tokenize_sample)
    - Iteration (__iter__)
    - Dataset loading (load_dataset class method)
    
    Subclasses must implement:
    - _pre_process_sample: Extract and format text from a sample dict
    """
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        streaming: Optional[bool] = None,
        timeout: Optional[int] = None,
        num_retries: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_dataset_workers: Optional[int] = None,
        **kwargs  # Allow subclasses to pass additional parameters
    ):   
        """
        Initialize the dataset.
        
        Args:
            dataset: The HuggingFace dataset (streaming or regular)
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length (defaults to data_config.max_length)
            batch_size: Batch size for data loading (defaults to data_config.batch_size)
            streaming: Whether to use streaming mode (defaults to data_config.streaming)
            timeout: Timeout for dataset downloads (defaults to data_config.timeout)
            num_retries: Number of retry attempts (defaults to data_config.num_retries)
            num_workers: Number of parallel workers (defaults to data_config.num_workers)
            num_dataset_workers: Number of dataset workers (defaults to data_config.num_dataset_workers)
            **kwargs: Additional parameters for subclass-specific configuration
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        # Use provided values or fall back to data_config defaults
        self.max_length = max_length if max_length is not None else data_config.max_length
        self.batch_size = batch_size if batch_size is not None else data_config.batch_size
        self.streaming = streaming if streaming is not None else data_config.streaming
        self.timeout = timeout if timeout is not None else data_config.timeout
        self.num_retries = num_retries if num_retries is not None else data_config.num_retries
        self.num_workers = num_workers if num_workers is not None else data_config.num_workers
        self.num_dataset_workers = num_dataset_workers if num_dataset_workers is not None else data_config.num_dataset_workers
        
        # Store any additional kwargs for subclass use
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def _pre_process_sample(self, sample: dict) -> Optional[str]:
        """
        Extract and preprocess text from a dataset sample.
        
        This method must be implemented by subclasses to handle dataset-specific
        text extraction logic (e.g., combining fields, filtering by language, etc.)
        
        Args:
            sample: A dictionary containing the dataset sample
            
        Returns:
            The preprocessed text string, or None if the sample should be skipped
        """
        pass

    def _tokenize_sample(self, text: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize text and create input/target sequences.
        
        This is common across all datasets and handles:
        - Encoding text to tokens
        - Truncation/padding to max_length
        - Clamping token IDs to valid vocabulary range
        - Creating shifted input/target pairs
        
        Args:
            text: The preprocessed text string
            
        Returns:
            Tuple of (input_ids, target_ids) tensors, or None if tokenization fails
        """
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
    
    def _process_sample(self, sample: dict) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a single sample: preprocess text, tokenize, and format.
        
        This combines the dataset-specific preprocessing with common tokenization.
        
        Returns:
            A tuple of (input_ids, target_ids) or None if sample is invalid
        """
        # Call subclass-specific preprocessing
        text = self._pre_process_sample(sample)
        
        # Skip if preprocessing returned None or empty text
        if not text or len(text.strip()) == 0:
            return None
        
        # Apply common tokenization
        return self._tokenize_sample(text)
    
    def __iter__(self):
        """
        Iterate over the dataset, yielding processed samples.
        
        This is called by PyTorch's DataLoader for IterableDataset.
        """
        for sample in self.dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                yield processed

    @staticmethod
    def load_dataset(
        dataset_name: DatasetName,
        data_subset_name: Optional[str],
        split: DatasetSplit = DatasetSplit.TRAIN,
        streaming: bool = True,
        num_retries: int = data_config.num_retries,
        timeout: int = data_config.timeout,
        max_samples: int = data_config.max_samples,
        use_val_split: bool = False,
        val_split_percentage: Optional[float] = 0.0,
    ):
        """
        Load a dataset from HuggingFace with retry logic and train/val splitting.
        
        Uses defaults from DataConfig unless explicitly overridden.
        
        This function includes retry logic to handle network timeouts and connection issues.
        
        Args:
            dataset_name: Name/path of the HuggingFace dataset (e.g., "PleIAs/SYNTH")
            data_subset_name: Optional dataset configuration name (e.g., "sample-10BT", "en", "default")
            split: Dataset split to load ("train", "validation", etc.)
            streaming: If True, stream the dataset (default from DataConfig)
            num_retries: Number of times to retry on failure (default from DataConfig)
            timeout: Timeout in seconds for download operations (default from DataConfig)
            max_samples: Maximum number of samples to load (default from DataConfig)
            use_val_split: If True, return the validation portion (last val_split_percentage)
                          If False, return the train portion (first 1-val_split_percentage)
            val_split_percentage: Percentage of data to use for validation (0.0 to 1.0)
            
        Returns:
            The loaded HuggingFace dataset (streaming or regular)
            
        Raises:
            FileNotFoundError: If dataset cannot be loaded after retries
        """
        # Use config defaults if not specified
        if streaming is None:
            streaming = data_config.streaming
        if num_retries is None:
            num_retries = data_config.num_retries
        if timeout is None:
            timeout = data_config.timeout
        if max_samples is None:
            max_samples = data_config.max_samples
        
        print(f"Loading {dataset_name} dataset from Hugging Face...")
        if data_subset_name:
            print(f"  Configuration: {data_subset_name}")
        print(f"  Split: {split}")
        if use_val_split:
            print(f"  Using VALIDATION portion (last {val_split_percentage*100:.0f}% of data)")
        else:
            print(f"  Using TRAINING portion (first {(1-val_split_percentage)*100:.0f}% of data)")
        print(f"  Streaming: {streaming} (recommended for large datasets)")
        print(f"  Max samples: {max_samples if max_samples else 'all'}")
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
                # Default to streaming=True as it doesn't require downloading everything upfront
                print("Connecting to Hugging Face Hub...")
                
                # Build load_dataset arguments
                # Convert enum to string value for HuggingFace API
                split_str = split.value if isinstance(split, DatasetSplit) else split
                load_args = {
                    "path": dataset_name,
                    "split": split_str,
                    "streaming": streaming,
                }
                
                # Add name parameter if specified
                if data_subset_name is not None:
                    load_args["name"] = data_subset_name
                
                # Add download_config for non-streaming datasets
                if not streaming:
                    load_args["download_config"] = {
                        "timeout": timeout,
                        "num_proc": 1,  # Reduce parallel downloads to avoid timeouts
                    }
                
                dataset = load_dataset(**load_args)
                
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
                    print("5. Check if you can access https://huggingface.co/datasets/{dataset_name}")
                    print("=" * 60)
                    raise FileNotFoundError(
                        f"Could not load {dataset_name} dataset after {num_retries} attempts. "
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
        
        return dataset  