import torch
from typing import Optional
from dataset_loaders.datasetprep import DatasetPrep, DatasetLang
from config import DataConfig

# Custom config overrides (optional)
# Uncomment and modify these to override defaults from config.py
# CUSTOM_MAX_LENGTH = 512
# CUSTOM_TEXT_FIELD = "text"

class RedPajamaDataset(DatasetPrep):
    """
    A PyTorch IterableDataset class for the RedPajama dataset.
    
    Inherits common tokenization and iteration logic from DatasetPrep.
    Implements RedPajama-specific text preprocessing.
    
    RedPajama is a large-scale open dataset for training large language models,
    containing data from multiple sources including CommonCrawl, C4, GitHub, etc.
    """
    
    def __init__(
        self, 
        dataset, 
        tokenizer, 
        max_length: int = 1024, 
        text_field: str = "text",
        language: DatasetLang = DatasetLang.ENGLISH
    ):
        """
        Initialize the RedPajama dataset.
        
        Args:
            dataset: The HuggingFace dataset (streaming or regular)
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length
            text_field: Which field contains the text (default: "text")
            language: Language filter for the dataset
        """
        # Call parent constructor
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            text_field=text_field,
            language=language
        )
    
    def _pre_process_sample(self, sample: dict) -> Optional[str]:
        """
        Extract and preprocess text from a RedPajama sample.
        
        RedPajama samples have a text field and metadata about the source.
        This method can be extended to add source-specific filtering or
        preprocessing logic.
        
        Args:
            sample: A dictionary containing the RedPajama sample
            
        Returns:
            The preprocessed text string, or None if the sample should be skipped
        """
        # Get the text from the specified field
        text = sample.get(self.text_field, "")
        
        # Skip empty samples
        if not text or len(text.strip()) == 0:
            return None
        
        # Optional: Filter by metadata (e.g., source type)
        # meta = sample.get('meta', {})
        # if meta.get('redpajama_set_name') == 'RedPajamaGithub':
        #     # Could add special handling for code
        #     pass
        
        return text

