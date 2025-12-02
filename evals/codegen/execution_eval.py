"""
Execution-based Evaluation

Tests if generated code actually runs through hierarchical checks:
1. Parses (valid Python syntax)
2. Instantiates (nn.Module can be created)
3. Forward pass works
4. Backward pass works
5. Trains stably (no NaN after N steps)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.rl.network_gym import CodeGenEnv, ExecutionResult
from training.codegen.config import EXAMPLE_CHALLENGES
from utils.device import get_best_device


@dataclass
class ExecutionMetrics:
    """Metrics from execution-based evaluation."""
    total: int
    parse_rate: float
    instantiate_rate: float
    forward_rate: float
    backward_rate: float
    stable_rate: float
    avg_reward: float
    
    def __str__(self):
        return f"""Execution Evaluation Results:
  Total samples: {self.total}
  Parse rate:       {self.parse_rate:.1%}
  Instantiate rate: {self.instantiate_rate:.1%}
  Forward rate:     {self.forward_rate:.1%}
  Backward rate:    {self.backward_rate:.1%}
  Stable rate:      {self.stable_rate:.1%}
  Average reward:   {self.avg_reward:.3f}"""


class ExecutionEvaluator:
    """
    Evaluate code generation model using execution feedback.
    
    Uses CodeGenEnv to test if generated code actually works.
    """
    
    def __init__(
        self,
        model_path: str,
        challenges_path: str = None,
        timeout: float = 10.0,
    ):
        self.device = get_best_device()
        
        # Load model (float32 for stable generation)
        print(f"Loading model: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load challenges
        self.challenges = self._load_challenges(challenges_path)
        
        # Create environment
        self.env = CodeGenEnv(
            challenges=self.challenges,
            tokenizer=self.tokenizer,
            timeout=timeout,
        )
    
    def _load_challenges(self, path: str | None) -> list[dict]:
        """Load challenges from JSONL or use examples."""
        if path and Path(path).exists():
            challenges = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        challenges.append(json.loads(line))
            return challenges
        return EXAMPLE_CHALLENGES
    
    def evaluate(
        self,
        max_samples: int = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> ExecutionMetrics:
        """
        Run execution-based evaluation.
        
        Args:
            max_samples: Limit number of challenges to test
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            
        Returns:
            ExecutionMetrics with pass rates
        """
        challenges = self.challenges[:max_samples] if max_samples else self.challenges
        
        results = []
        rewards = []
        
        print(f"\nEvaluating on {len(challenges)} challenges...")
        
        for challenge in tqdm(challenges, desc="Execution eval"):
            prompt = challenge["prompt"]
            
            # Format and generate
            formatted = f"# Task: {prompt}\n\nimport torch\nimport torch.nn as nn\n\nclass "
            
            input_ids = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                # Use greedy decoding to avoid NaN issues with FP16
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy - more stable
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract code
            if "class " in generated:
                code = generated[generated.find("class "):]
            else:
                code = generated
            
            # Execute and collect result
            result = self.env._execute_code(code)
            reward = self.env._compute_reward(result)
            
            results.append(result)
            rewards.append(reward)
        
        # Compute metrics
        total = len(results)
        metrics = ExecutionMetrics(
            total=total,
            parse_rate=sum(r.parses for r in results) / total,
            instantiate_rate=sum(r.instantiates for r in results) / total,
            forward_rate=sum(r.forward_works for r in results) / total,
            backward_rate=sum(r.backward_works for r in results) / total,
            stable_rate=sum(r.trains_stable for r in results) / total,
            avg_reward=sum(rewards) / total,
        )
        
        return metrics
    
    def evaluate_single(self, prompt: str, verbose: bool = True) -> tuple[str, ExecutionResult, float]:
        """Evaluate a single prompt."""
        formatted = f"# Task: {prompt}\n\nimport torch\nimport torch.nn as nn\n\nclass "
        
        input_ids = self.tokenizer(
            formatted,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,  # Greedy for stability
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "class " in generated:
            code = generated[generated.find("class "):]
        else:
            code = generated
        
        result = self.env._execute_code(code)
        reward = self.env._compute_reward(result)
        
        if verbose:
            print(f"\nPrompt: {prompt}")
            print(f"\nGenerated code:\n{code[:500]}...")
            print(f"\nExecution result:")
            print(f"  Parses: {result.parses}")
            print(f"  Instantiates: {result.instantiates}")
            print(f"  Forward: {result.forward_works}")
            print(f"  Backward: {result.backward_works}")
            print(f"  Stable: {result.trains_stable}")
            print(f"  Reward: {reward:.3f}")
            if result.error_msg:
                print(f"  Error: {result.error_msg[:200]}")
        
        return code, result, reward

