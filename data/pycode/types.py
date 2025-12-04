"""
Types and enums for code challenge data.
"""

from enum import Enum


class Source(str, Enum):
    """
    Source of a code challenge.
    
    Tracks where the training example came from.
    Use specific names that clearly identify the source.
    """
    
    # Hand-written examples
    MANUAL = "manual"
    
    # GitHub: HuggingFace repos
    GITHUB_HF_TIMM = "github:huggingface/pytorch-image-models"
    GITHUB_HF_TRANSFORMERS = "github:huggingface/transformers"
    GITHUB_HF_DIFFUSERS = "github:huggingface/diffusers"
    GITHUB_HF_ACCELERATE = "github:huggingface/accelerate"
    
    # GitHub: PyTorch repos
    GITHUB_PYTORCH_VISION = "github:pytorch/vision"
    GITHUB_PYTORCH_AUDIO = "github:pytorch/audio"
    GITHUB_PYTORCH_TEXT = "github:pytorch/text"
    
    # GitHub: Meta/Facebook repos
    GITHUB_META_DETECTRON2 = "github:facebookresearch/detectron2"
    GITHUB_META_FAIRSEQ = "github:facebookresearch/fairseq"
    
    # GitHub: Other notable repos
    GITHUB_OPENAI_WHISPER = "github:openai/whisper"
    GITHUB_LUCIDRAINS_VIT = "github:lucidrains/vit-pytorch"
    
    # HuggingFace Hub datasets
    HF_DATASET = "huggingface:dataset"
    
    # Other
    SYNTHETIC = "synthetic"  # Generated/augmented examples
    UNKNOWN = "unknown"
    
    @classmethod
    def values(cls) -> list[str]:
        """Get all valid source values."""
        return [s.value for s in cls]


# Lookup table: repo string -> Source enum
# Use exact repo paths for matching
REPO_TO_SOURCE: dict[str, Source] = {
    # HuggingFace repos
    "huggingface/pytorch-image-models": Source.GITHUB_HF_TIMM,
    "huggingface/transformers": Source.GITHUB_HF_TRANSFORMERS,
    "huggingface/diffusers": Source.GITHUB_HF_DIFFUSERS,
    "huggingface/accelerate": Source.GITHUB_HF_ACCELERATE,
    
    # PyTorch repos
    "pytorch/vision": Source.GITHUB_PYTORCH_VISION,
    "pytorch/audio": Source.GITHUB_PYTORCH_AUDIO,
    "pytorch/text": Source.GITHUB_PYTORCH_TEXT,
    
    # Meta repos
    "facebookresearch/detectron2": Source.GITHUB_META_DETECTRON2,
    "facebookresearch/fairseq": Source.GITHUB_META_FAIRSEQ,
    
    # Other repos
    "openai/whisper": Source.GITHUB_OPENAI_WHISPER,
    "lucidrains/vit-pytorch": Source.GITHUB_LUCIDRAINS_VIT,
}


def get_source(repo: str) -> str:
    """
    Get source string from a GitHub repo.
    
    Args:
        repo: GitHub repo like "huggingface/pytorch-image-models"
        
    Returns:
        Source enum value or formatted "github:{repo}" string
    """
    # Normalize repo string
    repo = repo.lower().strip()
    if repo.startswith("https://github.com/"):
        repo = repo.replace("https://github.com/", "")
    if repo.endswith(".git"):
        repo = repo[:-4]
    
    # Lookup in table
    source = REPO_TO_SOURCE.get(repo)
    if source:
        return source.value
    
    # Unknown repo - return formatted string
    return f"github:{repo}"


# For convenience
VALID_SOURCES = Source.values()

