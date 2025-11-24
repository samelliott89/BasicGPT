import torch
from typing import Optional
from datasets.datasetprep import DatasetPrep, DatasetLang

class FineWebDataset(DatasetPrep):
    """
    A PyTorch IterableDataset class for the FineWeb dataset.
    
    Inherits common tokenization and iteration logic from DatasetPrep.
    Implements FineWeb-specific text preprocessing.
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
        Initialize the FineWeb dataset.
        
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
        Extract and preprocess text from a FineWeb sample.
        
        FineWeb samples typically have a simple structure with a text field.
        This method can be extended to add FineWeb-specific filtering or
        preprocessing logic.
        
        Args:
            sample: A dictionary containing the FineWeb sample
            
        Returns:
            The preprocessed text string, or None if the sample should be skipped
        """
        # Get the text from the specified field
        text = sample.get(self.text_field, "")
        
        # Skip empty samples
        if not text or len(text.strip()) == 0:
            return None
        
        # Optional: Add language filtering if the dataset has a language field
        if hasattr(self, 'language') and 'language' in sample:
            sample_language = sample.get('language', '')
            if sample_language and sample_language != self.language:
                return None
        
        return text
    
    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        tokenizer,
        max_length: int = 1024,
        text_field: str = "text",
        language: DatasetLang = DatasetLang.ENGLISH,
        split: str = "train",
        streaming: bool = True,
        num_retries: int = 3,
        timeout: int = 300,
        max_samples: Optional[int] = None,
        use_val_split: bool = False,
        val_split_percentage: float = 0.1,
    ):
        """
        Load FineWeb dataset from HuggingFace and create a FineWebDataset instance.
        
        This is a convenience class method that combines dataset loading and
        instantiation in one step.
        
        Args:
            dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb")
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length
            text_field: Which field contains the text
            language: Language filter
            split: Dataset split to load
            streaming: Whether to stream the dataset
            num_retries: Number of retry attempts
            timeout: Timeout for downloads
            max_samples: Maximum number of samples
            use_val_split: Whether to use validation split
            val_split_percentage: Percentage for validation split
            
        Returns:
            FineWebDataset instance ready for training
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
        
        # Create and return the FineWebDataset instance
        return cls(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            text_field=text_field,
            language=language
        )