import torch
from typing import Optional

from data.loaders.datasetprep import DatasetPrep, DatasetLang
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
        max_length: Optional[int] = None,
        text_field: Optional[str] = None,
        include_reasoning: bool = False,
        filter_english_only: bool = True,
        **kwargs
    ):
        """
        Initialize the SYNTH iterable dataset.
        
        Args:
            dataset: The Hugging Face IterableDataset (streaming mode)
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length (defaults to data_config.max_length)
            text_field: Which field from the dataset to use as text (defaults to "synthetic_answer")
            include_reasoning: If True, include reasoning steps in training data
            filter_english_only: If True, filter for English-only samples
            **kwargs: Additional config parameters to pass to parent class
        """
        # Set dataset-specific defaults
        if text_field is None:
            text_field = "synthetic_answer"
        
        # Call parent constructor with common config parameters
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs
        )
        
        # Set SYNTH-specific attributes
        self.text_field = text_field
        self.include_reasoning = include_reasoning
        self.filter_english_only = filter_english_only
    
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