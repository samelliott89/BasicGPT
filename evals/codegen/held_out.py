"""
Held-Out Test Set Evaluation

Evaluates on challenges similar to training but not seen during training.
Split your challenges.jsonl into train/test for this.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.rl.network_gym import CodeGenEnv
from utils.device import get_best_device


@dataclass
class HeldOutMetrics:
    """Metrics from held-out evaluation."""
    total: int
    parse_rate: float
    instantiate_rate: float
    forward_rate: float
    backward_rate: float
    stable_rate: float
    avg_reward: float
    
    def __str__(self):
        return f"""Held-Out Evaluation Results:
  Total samples: {self.total}
  Parse rate:       {self.parse_rate:.1%}
  Instantiate rate: {self.instantiate_rate:.1%}
  Forward rate:     {self.forward_rate:.1%}
  Backward rate:    {self.backward_rate:.1%}
  Stable rate:      {self.stable_rate:.1%}
  Average reward:   {self.avg_reward:.3f}"""


def split_challenges(
    input_path: str,
    train_path: str,
    test_path: str,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Split challenges.jsonl into train and test sets.
    
    Args:
        input_path: Path to challenges.jsonl
        train_path: Output path for training set
        test_path: Output path for test set
        test_ratio: Fraction to hold out for testing
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    with open(input_path) as f:
        challenges = [json.loads(line) for line in f if line.strip()]
    
    random.shuffle(challenges)
    
    split_idx = int(len(challenges) * (1 - test_ratio))
    train = challenges[:split_idx]
    test = challenges[split_idx:]
    
    with open(train_path, "w") as f:
        for c in train:
            f.write(json.dumps(c) + "\n")
    
    with open(test_path, "w") as f:
        for c in test:
            f.write(json.dumps(c) + "\n")
    
    print(f"✓ Split {len(challenges)} challenges:")
    print(f"  Train: {len(train)} ({train_path})")
    print(f"  Test:  {len(test)} ({test_path})")
    
    return train, test


class HeldOutEvaluator:
    """
    Evaluate on held-out test set.
    
    Uses execution-based evaluation on challenges not seen during training.
    """
    
    def __init__(
        self,
        model_path: str,
        test_path: str,
        timeout: float = 10.0,
    ):
        self.device = get_best_device()
        
        print(f"Loading model: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load test set
        self.test_set = []
        if Path(test_path).exists():
            with open(test_path) as f:
                for line in f:
                    if line.strip():
                        self.test_set.append(json.loads(line))
            print(f"✓ Loaded {len(self.test_set)} held-out challenges")
        else:
            print(f"⚠️  Test set not found: {test_path}")
            print("   Run split_challenges() first to create train/test split")
        
        # Environment for execution testing
        self.env = CodeGenEnv(
            challenges=self.test_set,
            tokenizer=self.tokenizer,
            timeout=timeout,
        )
    
    def evaluate(
        self,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> HeldOutMetrics:
        """
        Evaluate on held-out test set.
        
        Returns:
            HeldOutMetrics with pass rates
        """
        if not self.test_set:
            return HeldOutMetrics(0, 0, 0, 0, 0, 0, 0)
        
        results = []
        rewards = []
        
        print(f"\nEvaluating on {len(self.test_set)} held-out challenges...")
        
        for challenge in tqdm(self.test_set, desc="Held-out eval"):
            prompt = challenge["prompt"]
            
            # Generate
            formatted = f"# Task: {prompt}\n\nimport torch\nimport torch.nn as nn\n\nclass "
            
            input_ids = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "class " in generated:
                code = generated[generated.find("class "):]
            else:
                code = generated
            
            # Execute
            result = self.env._execute_code(code)
            reward = self.env._compute_reward(result)
            
            results.append(result)
            rewards.append(reward)
        
        total = len(results)
        return HeldOutMetrics(
            total=total,
            parse_rate=sum(r.parses for r in results) / total,
            instantiate_rate=sum(r.instantiates for r in results) / total,
            forward_rate=sum(r.forward_works for r in results) / total,
            backward_rate=sum(r.backward_works for r in results) / total,
            stable_rate=sum(r.trains_stable for r in results) / total,
            avg_reward=sum(rewards) / total,
        )

