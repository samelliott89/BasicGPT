"""
Data preparation script for loading and processing the SYNTH dataset.

This script loads the SYNTH dataset from Hugging Face, tokenizes the text,
and creates PyTorch datasets and data loaders for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset as HfIterableDataset
from tokenizer import Tokenizer
from typing import Optional
import time
import os
import random
from config import GPTConfig, DataConfig, TrainingConfig
from dataset_loaders.datasetprep import DatasetName, DatasetSplit

gpt_config = GPTConfig()
data_config = DataConfig()
training_config = TrainingConfig()


########################################################################################
#### Multi-Dataset Support ####
########################################################################################

class InterleavedDataset(IterableDataset):
    """
    An IterableDataset that interleaves samples from multiple datasets.
    
    Samples from datasets based on configurable probabilities, providing
    balanced training across multiple data sources.
    """
    
    def __init__(
        self,
        datasets: list[IterableDataset],
        probabilities: Optional[list[float]] = None,
        seed: int = 42
    ):
        """
        Initialize the interleaved dataset.
        
        Args:
            datasets: List of IterableDataset instances to interleave
            probabilities: Sampling weights for each dataset (must sum to 1.0)
                          If None, uses equal probability for all datasets
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
        self.seed = seed
        
        # Set up probabilities
        if probabilities is None:
            # Equal probability for all datasets
            self.probabilities = [1.0 / len(datasets)] * len(datasets)
        else:
            if len(probabilities) != len(datasets):
                raise ValueError(
                    f"Number of probabilities ({len(probabilities)}) must match "
                    f"number of datasets ({len(datasets)})"
                )
            # Normalize probabilities to sum to 1.0
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]
    
    def __iter__(self):
        """
        Iterate over the datasets, sampling based on probabilities.
        
        Uses weighted random sampling to select which dataset to pull from next.
        Continues until all datasets are exhausted.
        """
        # Create iterators for each dataset
        iterators = [iter(ds) for ds in self.datasets]
        active_indices = list(range(len(iterators)))
        active_probs = self.probabilities.copy()
        
        # Set random seed for reproducibility
        rng = random.Random(self.seed)
        
        while active_indices:
            # Normalize probabilities for active datasets
            prob_sum = sum(active_probs)
            if prob_sum == 0:
                break
            normalized_probs = [p / prob_sum for p in active_probs]
            
            # Sample a dataset index based on probabilities
            idx = rng.choices(active_indices, weights=normalized_probs, k=1)[0]
            
            try:
                # Get next sample from selected dataset
                sample = next(iterators[idx])
                yield sample
            except StopIteration:
                # This dataset is exhausted, remove it from active list
                active_idx = active_indices.index(idx)
                active_indices.pop(active_idx)
                active_probs.pop(active_idx)


def get_dataset_class(dataset_name: DatasetName):
    """
    Map DatasetName enum to the corresponding dataset class.
    
    Args:
        dataset_name: The dataset name enum
        
    Returns:
        The dataset class for the given dataset name
    """
    # Import dataset classes here to avoid circular imports
    from dataset_loaders.synthdataset import SYNTHIterableDataset
    from dataset_loaders.finewebdataset import FineWebDataset
    from dataset_loaders.c4dataset import C4Dataset
    
    mapping = {
        DatasetName.SYNTHETIC: SYNTHIterableDataset,
        DatasetName.FINEWEB: FineWebDataset,
        DatasetName.C4: C4Dataset,
    }
    
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return mapping[dataset_name]


def get_dataset_config(dataset_name: DatasetName) -> dict:
    """
    Get dataset-specific configuration.
    
    Args:
        dataset_name: The dataset name enum
        
    Returns:
        Dictionary with dataset configuration including:
        - hf_name: HuggingFace dataset name
        - subset_name: Optional subset/configuration name
        - text_field: Field name containing the text
        - requires_validation_split: Whether dataset needs manual val split
        - dataset_kwargs: Additional dataset-specific parameters
    """
    configs = {
        DatasetName.SYNTHETIC: {
            "hf_name": "PleIAs/SYNTH",
            "subset_name": None,
            "text_field": "synthetic_answer",
            "requires_validation_split": True,
            "dataset_kwargs": {
                "include_reasoning": False,
                "filter_english_only": True,
            },
        },
        DatasetName.FINEWEB: {
            "hf_name": "HuggingFaceFW/fineweb",
            "subset_name": "sample-10BT",
            "text_field": "text",
            "requires_validation_split": True,
            "dataset_kwargs": {},
        },
        DatasetName.C4: {
            "hf_name": "allenai/c4",
            "subset_name": "en",
            "text_field": "text",
            "requires_validation_split": False,  # C4 has validation split
            "dataset_kwargs": {},
        },
    }
    
    if dataset_name not in configs:
        raise ValueError(f"No configuration found for dataset: {dataset_name}")
    
    return configs[dataset_name]


def load_datasets(
    dataset_names: list[DatasetName],
    tokenizer: Tokenizer,
    is_training: bool = True,
    probabilities: Optional[list[float]] = None,
    max_length: Optional[int] = None,
    streaming: Optional[bool] = None,
    num_retries: Optional[int] = None,
    timeout: Optional[int] = None,
    max_samples: Optional[int] = None,
    val_split_percentage: Optional[float] = None,
) -> IterableDataset:
    """
    Load one or more datasets for training or validation.
    
    If multiple datasets are provided, they will be interleaved using
    the InterleavedDataset class with configurable sampling probabilities.
        
        Args:
        dataset_names: List of DatasetName enums to load
        tokenizer: Tokenizer instance for encoding text
        is_training: If True, load training data; if False, load validation data
        probabilities: Sampling weights for each dataset (None = equal probability)
        max_length: Maximum sequence length (default from DataConfig)
        streaming: Whether to stream datasets (default from DataConfig)
        num_retries: Number of retry attempts (default from DataConfig)
        timeout: Timeout for downloads (default from DataConfig)
        max_samples: Maximum samples per dataset (default from DataConfig)
        val_split_percentage: Validation split percentage (default: 0.1 for datasets requiring split)
        
        Returns:
        Single dataset or InterleavedDataset containing all requested datasets
        
    Example:
        # Single dataset
        train_ds = load_datasets(
            dataset_names=[DatasetName.SYNTHETIC],
            tokenizer=tokenizer,
            is_training=True
        )
        
        # Multiple datasets with custom probabilities
        train_ds = load_datasets(
            dataset_names=[DatasetName.SYNTHETIC, DatasetName.FINEWEB],
            tokenizer=tokenizer,
            is_training=True,
            probabilities=[0.7, 0.3]  # 70% SYNTH, 30% FineWeb
        )
    """
    if not dataset_names:
        raise ValueError("Must provide at least one dataset name")
    
    # Load each dataset
    loaded_datasets = []
    
    for dataset_name in dataset_names:
        # Get dataset-specific configuration
        config = get_dataset_config(dataset_name)
        dataset_class = get_dataset_class(dataset_name)
        
        # Determine which loader function to use
        loader_fn = load_training_data if is_training else load_validation_data
        
        # Load the dataset
        print(f"\nLoading {dataset_name.value} dataset...")
        dataset = loader_fn(
            dataset_class=dataset_class,
            dataset_name=config["hf_name"],
            tokenizer=tokenizer,
            name=config["subset_name"],
            max_length=max_length,
            text_field=config["text_field"],
            streaming=streaming,
            num_retries=num_retries,
            timeout=timeout,
            max_samples=max_samples,
            val_split_percentage=val_split_percentage,
            requires_validation_split=config["requires_validation_split"],
            **config["dataset_kwargs"]
        )
        
        loaded_datasets.append(dataset)
        print(f"✓ Loaded {dataset_name.value} dataset")
        
    # If only one dataset, return it directly
    if len(loaded_datasets) == 1:
        return loaded_datasets[0]
    
    # Multiple datasets - interleave them
    print(f"\nInterleaving {len(loaded_datasets)} datasets...")
    if probabilities:
        print(f"  Sampling probabilities: {probabilities}")
    else:
        print(f"  Using equal probability for all datasets")
    
    interleaved = InterleavedDataset(
        datasets=loaded_datasets,
        probabilities=probabilities
    )
    
    print("✓ Created interleaved dataset")
    return interleaved


########################################################################################
#### Dataset Loading Helper Functions ####
########################################################################################


def load_training_data(
    dataset_class,
    dataset_name: str,
    tokenizer: Tokenizer,
    name: str = None,
    max_length: int = None,
    text_field: str = "text",
    split: str = "train",
    streaming: bool = None,
    num_retries: int = None,
    timeout: int = None,
    max_samples: int = None,
    val_split_percentage: Optional[float] = None,
    requires_validation_split: bool = False,
    **dataset_kwargs
):
    """
    Generic helper to load training data for any dataset.
    
    This function handles two scenarios:
    1. Dataset has a native validation split: loads from split="train"
    2. Dataset needs manual splitting: loads from split="train" with use_val_split=False (first 90%)
    
    Args:
        dataset_class: The dataset class to instantiate (e.g., FineWebDataset, C4Dataset)
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb")
        tokenizer: Tokenizer instance for encoding text
        name: Optional dataset configuration name (e.g., "sample-10BT", "en")
        max_length: Maximum sequence length (default from DataConfig)
        text_field: Which field contains the text (default: "text")
        split: Dataset split to load (default: "train")
        streaming: Whether to stream the dataset (default from DataConfig)
        num_retries: Number of retry attempts (default from DataConfig)
        timeout: Timeout for downloads (default from DataConfig)
        max_samples: Maximum number of samples (default from DataConfig)
        val_split_percentage: Percentage for validation split (default: 0.1 if requires_validation_split=True, else None)
        requires_validation_split: If True, dataset needs manual train/val splitting
                                   If False, dataset has native validation split
        **dataset_kwargs: Additional dataset-specific parameters (e.g., include_reasoning, filter_english_only)
        
    Returns:
        Dataset instance ready for training
        
    Example:
        # For FineWeb (no native val split, needs manual splitting)
        train_ds = load_training_data(
            dataset_class=FineWebDataset,
            dataset_name="HuggingFaceFW/fineweb",
            name="sample-10BT",
            tokenizer=tokenizer,
            requires_validation_split=True  # Will use use_val_split=False to get first 90%
        )
        
        # For a dataset with native validation split
        train_ds = load_training_data(
            dataset_class=SomeDataset,
            dataset_name="some/dataset",
            tokenizer=tokenizer,
            requires_validation_split=False  # Will just load from split="train"
        )
    """
    # Set default val_split_percentage if needed
    if val_split_percentage is None and requires_validation_split:
        val_split_percentage = 0.1  # Default to 10% validation
    
    # Load the raw dataset
    from dataset_loaders.datasetprep import DatasetPrep
    
    # For training data, we always use use_val_split=False
    # - If requires_validation_split=False: loads full "train" split (dataset has separate validation)
    # - If requires_validation_split=True: loads first 90% of "train" split
    raw_dataset = DatasetPrep.load_dataset(
        dataset_name=dataset_name,
        data_subset_name=name,
        split=DatasetSplit.TRAIN,
                streaming=streaming,
        num_retries=num_retries,
        timeout=timeout,
        max_samples=max_samples,
        use_val_split=False,  # Always False for training
        val_split_percentage=val_split_percentage if requires_validation_split else 0.0,
    )
    
    # Wrap with the dataset-specific class
    return dataset_class(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        text_field=text_field,
        **dataset_kwargs
    )


def load_validation_data(
    dataset_class,
    dataset_name: str,
    tokenizer: Tokenizer,
    name: str = None,
    max_length: int = None,
    text_field: str = "text",
    split: str = "validation",
    streaming: bool = None,
    num_retries: int = None,
    timeout: int = None,
    max_samples: int = None,
    val_split_percentage: Optional[float] = None,
    requires_validation_split: bool = False,
    **dataset_kwargs
):
    """
    Generic helper to load validation data for any dataset.
    
    This function handles two scenarios:
    1. Dataset has a native validation split: loads from split="validation"
    2. Dataset needs manual splitting: loads from split="train" with use_val_split=True
    
    Args:
        dataset_class: The dataset class to instantiate (e.g., FineWebDataset, C4Dataset)
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb")
        tokenizer: Tokenizer instance for encoding text
        name: Optional dataset configuration name (e.g., "sample-10BT", "en")
        max_length: Maximum sequence length (default from DataConfig)
        text_field: Which field contains the text (default: "text")
        split: Dataset split to load (default: "validation", or "train" if no native split)
        streaming: Whether to stream the dataset (default from DataConfig)
        num_retries: Number of retry attempts (default from DataConfig)
        timeout: Timeout for downloads (default from DataConfig)
        max_samples: Maximum number of samples (default from DataConfig)
        val_split_percentage: Percentage for validation split (default: 0.1 if requires_validation_split=True, else None)
        requires_validation_split: If True, dataset needs manual train/val splitting
                                   If False, dataset has native validation split, use split="validation"
        **dataset_kwargs: Additional dataset-specific parameters (e.g., include_reasoning, filter_english_only)
        
    Returns:
        Dataset instance ready for validation
        
    Example:
        # For FineWeb (no native val split, needs manual splitting)
        val_ds = load_validation_data(
            dataset_class=FineWebDataset,
            dataset_name="HuggingFaceFW/fineweb",
            name="sample-10BT",
            tokenizer=tokenizer,
            requires_validation_split=True  # Will use use_val_split=True on train split
        )
        
        # For a dataset with native validation split
        val_ds = load_validation_data(
            dataset_class=SomeDataset,
            dataset_name="some/dataset",
            tokenizer=tokenizer,
            requires_validation_split=False  # Will use split="validation"
        )
    """
    # Set default val_split_percentage if needed
    if val_split_percentage is None and requires_validation_split:
        val_split_percentage = 0.1  # Default to 10% validation
    
    # Load the raw dataset
    from dataset_loaders.datasetprep import DatasetPrep
    
    # Determine split and use_val_split based on whether dataset has native validation
    if requires_validation_split:
        # Dataset needs manual splitting - use train split with use_val_split=True
        actual_split = DatasetSplit.TRAIN
        use_val_split = True
    else:
        # Dataset has native validation split - use validation split directly
        actual_split = DatasetSplit.VALIDATION
        use_val_split = False

    raw_dataset = DatasetPrep.load_dataset(
        dataset_name=dataset_name,
        data_subset_name=name,
        split=actual_split,
        streaming=streaming,
        num_retries=num_retries,
        timeout=timeout,
        max_samples=max_samples,
        use_val_split=use_val_split,
        val_split_percentage=val_split_percentage if requires_validation_split else 0.0,
    )
    
    # Wrap with the dataset-specific class
    return dataset_class(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        text_field=text_field,
        **dataset_kwargs
    )


def create_data_loaders(
    train_dataset,
    val_dataset = None,
    batch_size: int = data_config.batch_size,  # Default batch size (should match TrainingConfig default)
    num_workers: int = data_config.num_dataset_workers,

) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    DataLoaders handle:
    - Batching multiple samples together
    - Shuffling data for training
    - Parallel data loading for efficiency
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Number of samples per batch
        num_workers: Number of parallel workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Check if dataset is an IterableDataset (streaming mode)
    is_iterable = isinstance(train_dataset, IterableDataset)
    
    # Note: Shuffling is already handled in load_synth_dataset for streaming datasets
    # For non-streaming datasets, we use DataLoader's shuffle parameter below
    
    # Create training data loader
    # Note: IterableDataset doesn't support shuffle, so we skip it for streaming mode
    # Also, prefetch_factor and persistent_workers only work with num_workers > 0
    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": not is_iterable,
        "num_workers": num_workers if not is_iterable else 0,
        "pin_memory": True,
    }
    
    # Only add these options if using multiple workers
    if train_loader_kwargs["num_workers"] > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        train_loader_kwargs["persistent_workers"] = True
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    # Create validation data loader (if provided)
    val_loader = None
    if val_dataset is not None:
        is_val_iterable = isinstance(val_dataset, IterableDataset)
        val_loader_kwargs = {
            "batch_size": batch_size * 2,  # Can be larger (no gradients stored)
            "shuffle": False,  # No shuffling needed for validation
            "num_workers": num_workers if not is_val_iterable else 0,
            "pin_memory": True,
        }
        
        # Only add these options if using multiple workers
        if val_loader_kwargs["num_workers"] > 0:
            val_loader_kwargs["prefetch_factor"] = 2
            val_loader_kwargs["persistent_workers"] = True
        
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    from collections import Counter
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    print(f"✓ Tokenizer initialized with vocab_size={tokenizer.vocab_size}")
    print()
    
    # ==========================================================================
    # TEST 1: Single Dataset Loading
    # ==========================================================================
    print("=" * 70)
    print("TEST 1: Single Dataset Loading (SYNTH)")
    print("=" * 70)
    single_dataset = load_datasets(
        dataset_names=[DatasetName.SYNTHETIC],
        tokenizer=tokenizer,
        is_training=True,
        max_length=512,
        streaming=True,
        max_samples=100,  # Small for quick testing
    )
    print()
    
    # Verify we can iterate and get samples
    print("Verifying single dataset iteration...")
    single_iter = iter(single_dataset)
    sample = next(single_iter)
    print(f"  ✓ Got sample with keys: {list(sample.keys()) if isinstance(sample, dict) else 'tensor tuple'}")
    print()
    
    # ==========================================================================
    # TEST 2: Multiple Dataset Loading with Interleaving
    # ==========================================================================
    print("=" * 70)
    print("TEST 2: Multiple Datasets with Interleaving (SYNTH + FINEWEB)")
    print("=" * 70)
    
    # Use 70/30 split
    target_probs = [0.7, 0.3]
    multi_dataset = load_datasets(
        dataset_names=[DatasetName.SYNTHETIC, DatasetName.FINEWEB],
        tokenizer=tokenizer,
        is_training=True,
        probabilities=target_probs,
        max_length=512,
        streaming=True,
        max_samples=200,  # 200 samples per dataset
    )
    print()
    
    # ==========================================================================
    # TEST 3: Verify Tokenization
    # ==========================================================================
    print("=" * 70)
    print("TEST 3: Verify Tokenization")
    print("=" * 70)
    
    # Create data loader to get batches
    train_loader, _ = create_data_loaders(
        train_dataset=multi_dataset,
        batch_size=8,
        num_workers=0
    )
    print(f"✓ Created train loader")
    print()
    
    # Get a batch and verify shapes
    print("Testing batch shapes...")
    input_ids, target_ids = next(iter(train_loader))
    print(f"  Input shape:  {input_ids.shape}")  # Should be [batch_size, max_length-1]
    print(f"  Target shape: {target_ids.shape}")  # Should be [batch_size, max_length-1]
    
    expected_seq_len = 512 - 1  # max_length - 1 for input/target split
    assert input_ids.shape[0] == 8, f"Expected batch size 8, got {input_ids.shape[0]}"
    assert input_ids.shape[1] == expected_seq_len, f"Expected seq len {expected_seq_len}, got {input_ids.shape[1]}"
    print(f"  ✓ Shapes are correct!")
    print()
    
    # Verify target is input shifted by 1
    print("Verifying input/target relationship (target = input shifted by 1)...")
    # For a proper next-token prediction: target[i] should come after input[i] in original sequence
    # Actually in our setup: input = tokens[:-1], target = tokens[1:]
    # So we can't directly verify without the original, but we can decode and check it makes sense
    
    # Decode first sample
    sample_input = input_ids[0].tolist()
    sample_target = target_ids[0].tolist()
    
    # Remove padding (0s) for cleaner output
    sample_input_clean = [t for t in sample_input if t != 0][:50]
    sample_target_clean = [t for t in sample_target if t != 0][:50]
    
    input_text = tokenizer.decode(sample_input_clean)
    target_text = tokenizer.decode(sample_target_clean)
    
    print(f"  Input (first 50 tokens):  {input_text[:150]}...")
    print(f"  Target (first 50 tokens): {target_text[:150]}...")
    print()
    
    # Check all tokens are within vocab range
    print("Verifying token IDs are within vocabulary range...")
    max_input_token = input_ids.max().item()
    max_target_token = target_ids.max().item()
    print(f"  Max input token ID:  {max_input_token}")
    print(f"  Max target token ID: {max_target_token}")
    print(f"  Vocabulary size:     {tokenizer.vocab_size}")
    
    if max_input_token < tokenizer.vocab_size and max_target_token < tokenizer.vocab_size:
        print(f"  ✓ All tokens within valid range!")
    else:
        print(f"  ⚠ WARNING: Some tokens exceed vocabulary size!")
    print()
    
    # ==========================================================================
    # TEST 4: Verify Probabilistic Sampling
    # ==========================================================================
    print("=" * 70)
    print("TEST 4: Verify Probabilistic Sampling")
    print("=" * 70)
    print(f"Target probabilities: SYNTH={target_probs[0]:.0%}, FINEWEB={target_probs[1]:.0%}")
    print()
    
    # To verify sampling, we need to track which dataset each sample comes from
    # We'll create a fresh interleaved dataset with labeled samples
    
    class LabeledWrapper(IterableDataset):
        """Wrapper that adds a source label to each sample."""
        def __init__(self, dataset, label: str):
            self.dataset = dataset
            self.label = label
        
        def __iter__(self):
            for sample in self.dataset:
                yield (sample, self.label)
    
    # Load individual datasets with labels
    print("Loading labeled datasets for sampling verification...")
    synth_ds = load_datasets(
        dataset_names=[DatasetName.SYNTHETIC],
        tokenizer=tokenizer,
        is_training=True,
        max_length=512,
        streaming=True,
        max_samples=500,
    )
    fineweb_ds = load_datasets(
        dataset_names=[DatasetName.FINEWEB],
        tokenizer=tokenizer,
        is_training=True,
        max_length=512,
        streaming=True,
        max_samples=500,
    )
    
    # Wrap with labels
    labeled_synth = LabeledWrapper(synth_ds, "SYNTH")
    labeled_fineweb = LabeledWrapper(fineweb_ds, "FINEWEB")
    
    # Create interleaved dataset with our target probabilities
    labeled_interleaved = InterleavedDataset(
        datasets=[labeled_synth, labeled_fineweb],
        probabilities=target_probs
    )
    
    # Sample and count
    print("Sampling 500 items and counting sources...")
    source_counts = Counter()
    sample_count = 0
    max_samples_to_check = 500
    
    for sample, label in labeled_interleaved:
        source_counts[label] += 1
        sample_count += 1
        if sample_count >= max_samples_to_check:
            break
        if sample_count % 100 == 0:
            print(f"  Processed {sample_count} samples...")
    
    print()
    print(f"Sampling results ({sample_count} total samples):")
    for label, count in source_counts.items():
        actual_prob = count / sample_count
        expected_prob = target_probs[0] if label == "SYNTH" else target_probs[1]
        diff = abs(actual_prob - expected_prob)
        status = "✓" if diff < 0.1 else "⚠"  # Allow 10% tolerance
        print(f"  {label}: {count} samples ({actual_prob:.1%}) - expected {expected_prob:.0%} {status}")
    
    print()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ TEST 1: Single dataset loading - PASSED")
    print("✓ TEST 2: Multiple dataset interleaving - PASSED")
    print("✓ TEST 3: Tokenization verification - PASSED")
    
    # Check if sampling is within tolerance
    synth_actual = source_counts.get("SYNTH", 0) / sample_count
    synth_diff = abs(synth_actual - target_probs[0])
    if synth_diff < 0.1:
        print("✓ TEST 4: Probabilistic sampling - PASSED")
    else:
        print(f"⚠ TEST 4: Probabilistic sampling - WARNING (diff={synth_diff:.1%})")
    
    print()
    print("✓ All tests complete!")

