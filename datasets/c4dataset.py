import torch
from typing import Optional
from datasets.datasetprep import DatasetPrep, DatasetLang
from config import DataConfig

# Custom config overrides (optional)
# Uncomment and modify these to override defaults from config.py
# CUSTOM_MAX_LENGTH = 512
# CUSTOM_TEXT_FIELD = "text"

class C4Dataset(DatasetPrep):
    """
    A PyTorch IterableDataset class for the C4 (Colossal Clean Crawled Corpus) dataset.
    
    Inherits common tokenization and iteration logic from DatasetPrep.
    Implements C4-specific text preprocessing.
    
    C4 is a cleaned version of Common Crawl, used to train models like T5.
    It's available in multiple languages.
    """
    
    def __init__(
        self, 
        dataset, 
        tokenizer, 
        max_length: Optional[int] = None, 
        text_field: Optional[str] = None,
        language: Optional[DatasetLang] = None,
        **kwargs
    ):
        """
        Initialize the C4 dataset.
        
        Args:
            dataset: The HuggingFace dataset (streaming or regular)
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length (defaults to data_config.max_length)
            text_field: Which field contains the text (defaults to "text")
            language: Language filter for the dataset (defaults to ENGLISH)
            **kwargs: Additional config parameters to pass to parent class
        """
        # Set dataset-specific defaults
        if text_field is None:
            text_field = "text"
        if language is None:
            language = DatasetLang.ENGLISH
        
        # Call parent constructor
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs
        )
        
        # Set C4-specific attributes
        self.text_field = text_field
        self.language = language
    
    def _pre_process_sample(self, sample: dict) -> Optional[str]:
        """
        Extract and preprocess text from a C4 sample.
        
        C4 samples have a text field and URL metadata.
        This method can be extended to add C4-specific filtering or
        preprocessing logic.
        
        Args:
            sample: A dictionary containing the C4 sample
            
        Returns:
            The preprocessed text string, or None if the sample should be skipped
        """
        # Get the text from the specified field
        text = sample.get(self.text_field, "")
        
        # Skip empty samples
        if not text or len(text.strip()) == 0:
            return None
        
        # Optional: Filter by URL or timestamp
        # url = sample.get('url', '')
        # timestamp = sample.get('timestamp', '')
        
        return text

