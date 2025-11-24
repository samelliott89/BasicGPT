import torch
from typing import Optional
from datasets.datasetprep import DatasetPrep, DatasetLang
from config import DataConfig

# Custom config overrides (optional)
# Uncomment and modify these to override defaults from config.py
# CUSTOM_MAX_LENGTH = 512
# CUSTOM_TEXT_FIELD = "synthetic_answer"
# CUSTOM_INCLUDE_REASONING = True
# CUSTOM_FILTER_ENGLISH_ONLY = False

class SYNTHIterableDataset(DatasetPrep):
    """
    A PyTorch IterableDataset class for the SYNTH dataset.
    
    Inherits common tokenization and iteration logic from DatasetPrep.
    Implements SYNTH-specific text preprocessing including:
    - Combining query, reasoning, and answer fields
    - Language filtering
    - Reasoning inclusion logic
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 1024,
        text_field: str = "synthetic_answer",
        include_reasoning: bool = False,
        filter_english_only: bool = True
    ):
        """
        Initialize the SYNTH iterable dataset.
        
        Args:
            dataset: The Hugging Face IterableDataset (streaming mode)
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length (context window size)
            text_field: Which field from the dataset to use as text
            include_reasoning: If True, include reasoning steps in training data
            filter_english_only: If True, filter for English-only samples
        """
        # Call parent constructor
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            text_field=text_field,
            include_reasoning=include_reasoning,
            filter_english_only=filter_english_only
        )
    
    def _pre_process_sample(self, sample: dict) -> Optional[str]:
        """
        Extract and preprocess text from a SYNTH sample.
        
        SYNTH samples have multiple fields that can be combined:
        - query: The question or prompt
        - synthetic_reasoning: The reasoning steps (optional)
        - synthetic_answer: The answer
        
        This method combines these fields based on configuration and
        filters by language if requested.
        
        Args:
            sample: A dictionary containing the SYNTH sample
            
        Returns:
            The preprocessed text string, or None if the sample should be skipped
        """
        # Filter for English-only if enabled (using dataset's language field)
        if self.filter_english_only:
            sample_language = sample.get('language')
            if sample_language:
                # Dataset uses ISO language codes: 'en', 'de', 'fr', etc.
                # Only keep English samples
                if str(sample_language).lower() != 'en':
                    return None  # Skip non-English samples
        
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
        
        return text
    
    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str = "PleIAs/SYNTH",
        tokenizer = None,
        max_length: int = None,
        text_field: str = None,
        include_reasoning: bool = None,
        filter_english_only: bool = None,
        split: str = "train",
        streaming: bool = None,
        num_retries: int = None,
        timeout: int = None,
        max_samples: Optional[int] = None,
        use_val_split: bool = False,
        val_split_percentage: float = 0.1,
    ):
        """
        Load SYNTH dataset from HuggingFace and create a SYNTHIterableDataset instance.
        
        Uses defaults from DataConfig unless explicitly overridden.
        
        This is a convenience class method that combines dataset loading and
        instantiation in one step.
        
        Args:
            dataset_name: HuggingFace dataset name (default: "PleIAs/SYNTH")
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length (default from DataConfig)
            text_field: Which field to use as text source (default from DataConfig)
            include_reasoning: If True, include reasoning steps (default from DataConfig)
            filter_english_only: If True, filter for English-only samples (default from DataConfig)
            split: Dataset split to load (SYNTH only has "train")
            streaming: Whether to stream the dataset (default from DataConfig)
            num_retries: Number of retry attempts (default from DataConfig)
            timeout: Timeout for downloads (default from DataConfig)
            max_samples: Maximum number of samples (default from DataConfig)
            use_val_split: Whether to use validation split
            val_split_percentage: Percentage for validation split
            
        Returns:
            SYNTHIterableDataset instance ready for training
        """
        # Get defaults from DataConfig
        data_config = DataConfig()
        
        # Use config defaults if not specified
        if max_length is None:
            max_length = data_config.max_length
        if text_field is None:
            text_field = data_config.text_field
        if include_reasoning is None:
            include_reasoning = data_config.include_reasoning
        if filter_english_only is None:
            filter_english_only = data_config.filter_english_only
        # Load the raw dataset using the parent class method
        dataset = DatasetPrep.load_dataset(
            dataset_name=dataset_name,
            split=split,
            streaming=streaming,
            num_retries=num_retries,
            timeout=timeout,
            max_samples=max_samples,
            use_val_split=use_val_split,
            val_split_percentage=val_split_percentage,
        )
        
        # Create and return the SYNTHIterableDataset instance
        return cls(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            text_field=text_field,
            include_reasoning=include_reasoning,
            filter_english_only=filter_english_only
        )


def load_synth_dataset(
    tokenizer,
    max_length: int = None,
    split: str = "train",
    streaming: bool = None,
    text_field: str = None,
    include_reasoning: bool = None,
    filter_english_only: bool = None,
    num_retries: int = None,
    timeout: int = None,
    max_samples: Optional[int] = None,
    val_split_percentage: float = 0.1,
    use_val_split: bool = False
):
    """
    Legacy wrapper for loading SYNTH dataset. 
    
    Uses defaults from DataConfig unless explicitly overridden.
    
    This function is maintained for backward compatibility.
    New code should use SYNTHIterableDataset.from_huggingface() instead.
    
    Args:
        tokenizer: The Tokenizer instance to use
        max_length: Maximum sequence length (default from DataConfig)
        split: Dataset split to load (SYNTH only has "train")
        streaming: If True, stream the dataset (default from DataConfig)
        text_field: Which field to use as text source (default from DataConfig)
        include_reasoning: If True, include reasoning steps (default from DataConfig)
        filter_english_only: If True, only use English samples (default from DataConfig)
        num_retries: Number of times to retry on failure (default from DataConfig)
        timeout: Timeout in seconds for download operations (default from DataConfig)
        max_samples: Maximum number of samples to load (default from DataConfig)
        val_split_percentage: Percentage of data to use for validation (0.0 to 1.0)
        use_val_split: If True, return validation portion; False for training portion
        
    Returns:
        A SYNTHIterableDataset instance ready for training
    """
    # Simply delegate to the class method
    return SYNTHIterableDataset.from_huggingface(
        dataset_name="PleIAs/SYNTH",
        tokenizer=tokenizer,
        max_length=max_length,
        text_field=text_field,
        include_reasoning=include_reasoning,
        filter_english_only=filter_english_only,
        split=split,
        streaming=streaming,
        num_retries=num_retries,
        timeout=timeout,
        max_samples=max_samples,
        use_val_split=use_val_split,
        val_split_percentage=val_split_percentage,
    )