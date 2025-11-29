"""
Unified model and tokenizer loading.

Usage:
    from model import load_model, ModelType

    # HuggingFace
    model, tokenizer = load_model("gpt2", ModelType.HUGGINGFACE)

    # BasicGPT
    model, tokenizer = load_model("./checkpoints/best", ModelType.BASICGPT)

    # Auto-detect
    model, tokenizer = load_model("gpt2")  # -> HuggingFace
    model, tokenizer = load_model("./checkpoints/best")  # -> BasicGPT
"""

from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

from config import GenerationConfig
from model.types import ModelOutput
from utils.device import get_best_device

_gen_config = GenerationConfig()


class ModelType(str, Enum):
    """Supported model types."""

    HUGGINGFACE = "huggingface"
    BASICGPT = "basicgpt"
    AUTO = "auto"


# =============================================================================
# Unified Language Model
# =============================================================================


class LanguageModel:
    """
    Unified wrapper for language models.

    Handles both HuggingFace and BasicGPT models with the same interface.
    """

    def __init__(self, model, model_type: ModelType, config=None):
        self._model = model
        self._type = model_type
        self._config = config or getattr(model, "config", None)
        self._device = next(model.parameters()).device

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

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

    @property
    def model_type(self) -> ModelType:
        return self._type

    # -------------------------------------------------------------------------
    # PyTorch-like methods
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Forward pass through the model."""
        match self._type:
            case ModelType.HUGGINGFACE:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                return ModelOutput(
                    logits=outputs.logits,
                    loss=getattr(outputs, "loss", None),
                )

            case ModelType.BASICGPT:
                logits = self._model(input_ids)
                loss = None
                if labels is not None:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                return ModelOutput(logits=logits, loss=loss)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = _gen_config.max_new_tokens,
        temperature: float = _gen_config.temperature,
        top_k: int = _gen_config.top_k,
        top_p: float = _gen_config.top_p,
        repetition_penalty: float = _gen_config.repetition_penalty,
        **kwargs,
    ) -> torch.Tensor:
        """Generate new tokens."""
        if self._type == ModelType.HUGGINGFACE:
            # HuggingFace needs these params
            kwargs.setdefault("do_sample", True)
            pad_id = self._model.config.pad_token_id or self._model.config.eos_token_id
            kwargs.setdefault("pad_token_id", pad_id)
            # Don't pass repetition_penalty to HuggingFace (causes early stopping)
            return self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )

        # BasicGPT - pass all params including repetition_penalty
        return self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------

    def save(self, path: str):
        """Save model (HuggingFace uses save_pretrained, BasicGPT uses torch.save)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        match self._type:
            case ModelType.HUGGINGFACE:
                self._model.save_pretrained(path)
            case ModelType.BASICGPT:
                torch.save(
                    {"model_state_dict": self._model.state_dict(), "gpt_config": self._config},
                    path / "checkpoint.pt",
                )

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    @classmethod
    def load(
        cls, path: str, model_type: ModelType = ModelType.AUTO, device: torch.device = None
    ) -> "LanguageModel":
        """Load a model from path or HuggingFace name."""
        device = device or get_best_device()

        if model_type == ModelType.AUTO:
            model_type = _detect_type(path)

        match model_type:
            case ModelType.HUGGINGFACE:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(path)
                model.to(device)
                return cls(model, model_type, model.config)

            case ModelType.BASICGPT:
                from model.config import GPTConfig
                from model.gpt import GPT

                path = Path(path)
                checkpoint_file = path / "checkpoint.pt" if path.is_dir() else path
                checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

                config = checkpoint.get("gpt_config", GPTConfig())
                if "model_state_dict" in checkpoint:
                    emb = checkpoint["model_state_dict"].get("token_embedding.weight")
                    if emb is not None:
                        config.vocab_size = emb.shape[0]

                model = GPT(config)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                return cls(model, model_type, config)

            case _:
                raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Unified Tokenizer
# =============================================================================


class Tokenizer:
    """
    Unified wrapper for tokenizers.

    Handles both HuggingFace and tiktoken tokenizers with the same interface.
    """

    def __init__(self, tokenizer, tokenizer_type: ModelType):
        self._tokenizer = tokenizer
        self._type = tokenizer_type

        # Ensure HF has pad token
        if self._type == ModelType.HUGGINGFACE and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        match self._type:
            case ModelType.HUGGINGFACE:
                return self._tokenizer.decode(token_ids, skip_special_tokens=True)
            case ModelType.BASICGPT:
                filtered = [t for t in token_ids if t != 0]  # filter padding
                return self._tokenizer.decode(filtered)

    @property
    def vocab_size(self) -> int:
        match self._type:
            case ModelType.HUGGINGFACE:
                return len(self._tokenizer)
            case ModelType.BASICGPT:
                return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        match self._type:
            case ModelType.HUGGINGFACE:
                return self._tokenizer.pad_token_id
            case ModelType.BASICGPT:
                return 0

    @property
    def eos_token_id(self) -> int:
        match self._type:
            case ModelType.HUGGINGFACE:
                return self._tokenizer.eos_token_id
            case ModelType.BASICGPT:
                return 100257  # <|endoftext|> in cl100k_base

    @property
    def tokenizer(self):
        """Access underlying tokenizer."""
        return self._tokenizer

    def __call__(
        self,
        text: str | list[str],
        max_length: int = 512,
        padding: bool = False,  # Changed: no padding by default for generation
        truncation: bool = True,
        return_tensors: str = None,
    ) -> dict:
        """Tokenize text with padding/truncation (HuggingFace-compatible interface)."""
        if self._type == ModelType.HUGGINGFACE:
            return self._tokenizer(
                text,
                max_length=max_length,
                padding="max_length" if padding else False,
                truncation=truncation,
                return_tensors=return_tensors,
            )

        # BasicGPT/tiktoken
        texts = [text] if isinstance(text, str) else text
        all_ids, all_masks = [], []

        for t in texts:
            ids = self.encode(t)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding:
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            all_ids.append(ids)
            all_masks.append(mask)

        result = {"input_ids": all_ids, "attention_mask": all_masks}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"])
            result["attention_mask"] = torch.tensor(result["attention_mask"])
        return result

    def save(self, path: str):
        """Save tokenizer (HuggingFace only)."""
        if self._type == ModelType.HUGGINGFACE:
            self._tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, tokenizer_type: ModelType = ModelType.AUTO) -> "Tokenizer":
        """Load a tokenizer."""
        if tokenizer_type == ModelType.AUTO:
            tokenizer_type = _detect_type(path)

        match tokenizer_type:
            case ModelType.HUGGINGFACE:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(path)
                return cls(tokenizer, tokenizer_type)

            case ModelType.BASICGPT:
                from data.tokenizer import Tokenizer as TikTokenizer

                tokenizer = TikTokenizer()
                return cls(tokenizer, tokenizer_type)

            case _:
                raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


# =============================================================================
# Convenience functions
# =============================================================================


def load_model(
    path: str,
    model_type: ModelType | str = ModelType.AUTO,
    device: torch.device = None,
) -> tuple[LanguageModel, Tokenizer]:
    """Load a model and tokenizer."""
    device = device or get_best_device()

    # Convert string to enum if needed
    if isinstance(model_type, str):
        model_type = ModelType(model_type) if model_type != "auto" else ModelType.AUTO

    if model_type == ModelType.AUTO:
        model_type = _detect_type(path)

    model = LanguageModel.load(path, model_type, device)
    tokenizer = Tokenizer.load(path, model_type)

    return model, tokenizer


def _detect_type(path: str) -> ModelType:
    """Auto-detect model type from path."""
    p = Path(path)

    if p.suffix == ".pt":
        return ModelType.BASICGPT
    if p.exists() and (p.is_file() or (p / "checkpoint.pt").exists()):
        return ModelType.BASICGPT

    return ModelType.HUGGINGFACE
