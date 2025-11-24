    # if isinstance(dataset, HfIterableDataset) or streaming:
    #     print("Using IterableDataset for streaming mode...")
    #     dataset = SYNTHIterableDataset(
    #         dataset=dataset,
    #         tokenizer=tokenizer,
    #         max_length=max_length,
    #         text_field=text_field,
    #         include_reasoning=include_reasoning,
    #     )
    # else:
    #     print("Using regular Dataset (pre-tokenizing all samples)...")
    #     synth_dataset = SYNTHDataset(
    #         dataset=dataset,
    #         tokenizer=tokenizer,
    #         max_length=max_length,
    #         text_field=text_field
    #     )



    from datasets import DatasetPrep

class SYNTHIterableDataset(DatasetPrep):
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