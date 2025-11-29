"""
Model and tokenizer wrappers that implement the base interfaces.

These wrappers provide a unified interface for:
- HuggingFace models (GPT-2, etc.)
- BasicGPT custom models
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from config import GenerationConfig
from model.base import BaseLanguageModel, BaseTokenizer, ModelOutput
from utils.device import get_best_device

# Default generation config instance for default parameter values
_default_gen_config = GenerationConfig()


# ============================================================================
# Common Model Wrapper Base
# ============================================================================


class ModelWrapper(BaseLanguageModel):
    """
    Base class with shared functionality for model wrappers.

    Both HuggingFace and BasicGPT wrappers inherit from this.
    """

    def __init__(self, model, config=None):
        self._model = model
        self._config = config
        self._device = next(model.parameters()).device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        """Access underlying model."""
        return self._model

    def train(self):
        self._model.train()
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    def named_parameters(self):
        return self._model.named_parameters()

    def to(self, device):
        self._model.to(device)
        self._device = device
        return self

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        return self._model.load_state_dict(state_dict)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = _default_gen_config.max_new_tokens,
        temperature: float = _default_gen_config.temperature,
        top_k: int = _default_gen_config.top_k,
        top_p: float = _default_gen_config.top_p,
        repetition_penalty: float = _default_gen_config.repetition_penalty,
        **kwargs,
    ) -> torch.Tensor:
        """Generate new tokens. Override in subclass if needed."""
        return self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )


# ============================================================================
# HuggingFace Wrappers
# ============================================================================


class HuggingFaceModel(ModelWrapper):
    """
    Wrapper for HuggingFace causal language models.

    Usage:
        model = HuggingFaceModel.load("gpt2")
        model = HuggingFaceModel.load("gpt2-medium")
        model = HuggingFaceModel.load("./my-finetuned-model")
    """

    def __init__(self, model, model_name: str = None):
        super().__init__(model, config=model.config)
        self._model_name = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return ModelOutput(
            logits=outputs.logits,
            loss=outputs.loss if hasattr(outputs, "loss") else None,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = _default_gen_config.max_new_tokens,
        temperature: float = _default_gen_config.temperature,
        top_k: int = _default_gen_config.top_k,
        top_p: float = _default_gen_config.top_p,
        repetition_penalty: float = _default_gen_config.repetition_penalty,
        **kwargs,
    ) -> torch.Tensor:
        # HuggingFace requires do_sample and pad_token_id
        return self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self._model.config.pad_token_id,
            **kwargs,
        )

    def save(self, path: str):
        self._model.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "HuggingFaceModel":
        from transformers import AutoModelForCausalLM

        if device is None:
            device = get_best_device()

        model = AutoModelForCausalLM.from_pretrained(path)
        model.to(device)

        return cls(model, model_name=path)


class HuggingFaceTokenizer(BaseTokenizer):
    """
    Wrapper for HuggingFace tokenizers.

    Usage:
        tokenizer = HuggingFaceTokenizer.load("gpt2")
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @classmethod
    def load(cls, path: str) -> "HuggingFaceTokenizer":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(tokenizer)

    def save(self, path: str):
        self._tokenizer.save_pretrained(path)

    @property
    def tokenizer(self):
        """Access underlying HuggingFace tokenizer."""
        return self._tokenizer


# ============================================================================
# BasicGPT Wrappers
# ============================================================================


class BasicGPTModel(ModelWrapper):
    """
    Wrapper for BasicGPT custom models.

    Usage:
        model = BasicGPTModel.load("./checkpoints/best/checkpoint.pt")
    """

    def __init__(self, model, config=None):
        super().__init__(model, config=config or model.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        # BasicGPT forward returns logits directly
        logits = self._model(input_ids)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute loss, ignoring padding (-100) and pad_token (0)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(logits=logits, loss=loss)

    # generate() inherited from ModelWrapper - BasicGPT uses the default implementation

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "gpt_config": self._config,
            },
            path / "checkpoint.pt",
        )

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "BasicGPTModel":
        from model.config import GPTConfig
        from model.gpt import GPT

        if device is None:
            device = get_best_device()

        path = Path(path)

        # Handle both folder and file paths
        if path.is_dir():
            checkpoint_file = path / "checkpoint.pt"
        else:
            checkpoint_file = path

        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

        # Get config
        config = checkpoint.get("gpt_config")
        if config is None:
            config = GPTConfig()

        # Check vocab size from weights
        if "model_state_dict" in checkpoint:
            emb_weight = checkpoint["model_state_dict"].get("token_embedding.weight")
            if emb_weight is not None:
                config.vocab_size = emb_weight.shape[0]

        # Create and load model
        model = GPT(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return cls(model, config)


class BasicGPTTokenizer(BaseTokenizer):
    """
    Wrapper for BasicGPT's tiktoken-based tokenizer.

    Usage:
        tokenizer = BasicGPTTokenizer.load()
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        # tiktoken doesn't have native pad/eos, we use special token IDs
        self._pad_token_id = 0  # Use 0 as pad
        self._eos_token_id = 100257  # <|endoftext|> in cl100k_base

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        # Filter out padding
        filtered = [t for t in token_ids if t != self._pad_token_id]
        return self._tokenizer.decode(filtered)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @classmethod
    def load(cls, encoding_name: str = "cl100k_base") -> "BasicGPTTokenizer":
        from data.tokenizer import Tokenizer

        tokenizer = Tokenizer(encoding_name=encoding_name)
        return cls(tokenizer)

    @property
    def tokenizer(self):
        """Access underlying tiktoken tokenizer."""
        return self._tokenizer
