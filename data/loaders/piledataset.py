from data.loaders.datasetprep import DatasetPrep
from enums import DatasetLang

# Custom config overrides (optional)
# Uncomment and modify these to override defaults from config.py
# CUSTOM_MAX_LENGTH = 512
# CUSTOM_TEXT_FIELD = "text"


class PileDataset(DatasetPrep):
    """
    A PyTorch IterableDataset class for the Pile dataset.

    Inherits common tokenization and iteration logic from DatasetPrep.
    Implements Pile-specific text preprocessing.

    The Pile is a large, diverse, open-source language modeling dataset
    created by EleutherAI, consisting of 22 diverse high-quality subsets.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 1024,
        text_field: str = "text",
        language: DatasetLang = DatasetLang.ENGLISH,
    ):
        """
        Initialize the Pile dataset.

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
            language=language,
        )

    def _pre_process_sample(self, sample: dict) -> str | None:
        """
        Extract and preprocess text from a Pile sample.

        Pile samples have a text field and metadata about which subset they come from.
        This method can be extended to add subset-specific filtering or
        preprocessing logic.

        Args:
            sample: A dictionary containing the Pile sample

        Returns:
            The preprocessed text string, or None if the sample should be skipped
        """
        # Get the text from the specified field
        text = sample.get(self.text_field, "")

        # Skip empty samples
        if not text or len(text.strip()) == 0:
            return None

        # Optional: Filter by pile_set_name (subset)
        # pile_set = sample.get('meta', {}).get('pile_set_name', '')
        # if pile_set == 'Github':
        #     # Could add special handling for code
        #     pass

        return text
