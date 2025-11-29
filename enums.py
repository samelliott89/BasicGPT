"""
Enums for dataset configuration.

Separated from config.py and datasetprep.py to avoid circular imports.
"""

from enum import Enum


class DatasetName(Enum):
    """Available datasets for training."""

    SYNTHETIC = "PleIAs/SYNTH"
    FINEWEB = "HuggingFaceFW/fineweb"
    C4 = "allenai/c4"
    BOOKCORPUS = "bookcorpus/bookcorpus"
    REDPAJAMA = "togethercomputer/RedPajama-Data-1T"
    PILE = "EleutherAI/pile"


class DatasetLang(str, Enum):
    """String enum for dataset languages (Python 3.10 compatible)."""

    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    SPANISH = "es"


class DatasetSplit(Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
