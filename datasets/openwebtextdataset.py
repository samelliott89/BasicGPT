import torch
from typing import Optional
from datasets.datasetprep import DatasetPrep, DatasetLang
from config import DataConfig

# Custom config overrides (optional)
# Uncomment and modify these to override defaults from config.py
# CUSTOM_MAX_LENGTH = 512
# CUSTOM_TEXT_FIELD = "text"

class OpenWebTextDataset(DatasetPrep):
    """
    A PyTorch IterableDataset class for the OpenWebText dataset.
    
    Inherits common tokenization and iteration logic from DatasetPrep.
    Implements OpenWebText-specific text preprocessing.
    
    OpenWebText is an open-source recreation of the WebText dataset used to train GPT-2.
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
        Initialize the OpenWebText dataset.
        
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
        Extract and preprocess text from an OpenWebText sample.
        
        OpenWebText samples have a simple structure with a text field.
        This method can be extended to add OpenWebText-specific filtering or
        preprocessing logic.
        
        Args:
            sample: A dictionary containing the OpenWebText sample
            
        Returns:
            The preprocessed text string, or None if the sample should be skipped
        """
        # Get the text from the specified field
        text = sample.get(self.text_field, "")
        
        # Skip empty samples
        if not text or len(text.strip()) == 0:
            return None
        
        return text

