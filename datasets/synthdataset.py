import torch
from typing import Optional
from datasets.datasetprep import DatasetPrep, DatasetLang

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
        max_length: int = 1024,
        text_field: str = "synthetic_answer",
        include_reasoning: bool = False,
        filter_english_only: bool = True,
        split: str = "train",
        streaming: bool = True,
        num_retries: int = 3,
        timeout: int = 300,
        max_samples: Optional[int] = None,
        use_val_split: bool = False,
        val_split_percentage: float = 0.1,
    ):
        """
        Load SYNTH dataset from HuggingFace and create a SYNTHIterableDataset instance.
        
        This is a convenience class method that combines dataset loading and
        instantiation in one step.
        
        Args:
            dataset_name: HuggingFace dataset name (default: "PleIAs/SYNTH")
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length
            text_field: Which field to use as text source
            include_reasoning: If True, include reasoning steps in training data
            filter_english_only: If True, filter for English-only samples
            split: Dataset split to load (SYNTH only has "train")
            streaming: Whether to stream the dataset
            num_retries: Number of retry attempts
            timeout: Timeout for downloads
            max_samples: Maximum number of samples
            use_val_split: Whether to use validation split
            val_split_percentage: Percentage for validation split
            
        Returns:
            SYNTHIterableDataset instance ready for training
        """
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
    max_length: int = 1024,
    split: str = "train",
    streaming: bool = True,
    text_field: str = "synthetic_answer",
    include_reasoning: bool = False,
    filter_english_only: bool = True,
    num_retries: int = 3,
    timeout: int = 300,
    max_samples: Optional[int] = None,
    val_split_percentage: float = 0.1,
    use_val_split: bool = False
):
    """
    Legacy wrapper for loading SYNTH dataset. 
    
    This function is maintained for backward compatibility.
    New code should use SYNTHIterableDataset.from_huggingface() instead.
    
    Args:
        tokenizer: The Tokenizer instance to use
        max_length: Maximum sequence length
        split: Dataset split to load (SYNTH only has "train")
        streaming: If True, stream the dataset (recommended)
        text_field: Which field to use as text source
        include_reasoning: If True, include reasoning steps in training data
        filter_english_only: If True, only use English samples
        num_retries: Number of times to retry on failure
        timeout: Timeout in seconds for download operations
        max_samples: Maximum number of samples to load (None = all)
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