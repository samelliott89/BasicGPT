"""
Standard Code Generation Benchmarks

Integrates with:
- HumanEval: OpenAI's function completion benchmark
- DS-1000: Data science benchmark with PyTorch subset
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.device import get_best_device


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    benchmark: str
    total: int
    passed: int
    pass_rate: float
    details: list[dict] = None
    
    def __str__(self):
        return f"{self.benchmark}: {self.passed}/{self.total} ({self.pass_rate:.1%})"


class HumanEvalRunner:
    """
    Run HumanEval benchmark.
    
    HumanEval tests function completion - given a docstring and signature,
    complete the function body.
    
    Dataset: openai/human-eval (164 problems)
    Metric: pass@k (usually pass@1)
    """
    
    def __init__(self, model_path: str):
        self.device = get_best_device()
        
        print(f"Loading model for HumanEval: {model_path}")
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
    
    def _load_humaneval(self) -> list[dict]:
        """Load HumanEval dataset."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai/openai_humaneval", split="test")
            return list(dataset)
        except Exception as e:
            print(f"⚠️  Could not load HumanEval: {e}")
            print("   Install with: pip install datasets")
            return []
    
    def _extract_function(self, generated: str, entry_point: str) -> str:
        """Extract the function body from generated code."""
        # Find where the function ends (next def or class, or end)
        lines = generated.split("\n")
        in_function = False
        function_lines = []
        indent_level = None
        
        for line in lines:
            if line.strip().startswith(f"def {entry_point}"):
                in_function = True
                function_lines.append(line)
                continue
            
            if in_function:
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    # New top-level definition, stop
                    if line.strip().startswith("def ") or line.strip().startswith("class "):
                        break
                function_lines.append(line)
        
        return "\n".join(function_lines)
    
    def _run_test(self, code: str, test: str) -> bool:
        """Run a single test case."""
        full_code = code + "\n\n" + test
        
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                f.flush()
                
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    timeout=10,
                    text=True,
                )
                
                os.unlink(f.name)
                return result.returncode == 0
        except Exception:
            return False
    
    def evaluate(
        self,
        max_samples: int = None,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> BenchmarkResult:
        """
        Run HumanEval benchmark.
        
        Args:
            max_samples: Limit number of problems (None = all 164)
            temperature: Lower is more deterministic
            max_new_tokens: Max tokens to generate
            
        Returns:
            BenchmarkResult with pass@1
        """
        problems = self._load_humaneval()
        if not problems:
            return BenchmarkResult("HumanEval", 0, 0, 0.0)
        
        if max_samples:
            problems = problems[:max_samples]
        
        passed = 0
        details = []
        
        print(f"\nRunning HumanEval on {len(problems)} problems...")
        
        for problem in tqdm(problems, desc="HumanEval"):
            prompt = problem["prompt"]
            test = problem["test"]
            entry_point = problem["entry_point"]
            
            # Generate completion
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = self._extract_function(generated, entry_point)
            
            # Test
            success = self._run_test(code, test)
            if success:
                passed += 1
            
            details.append({
                "task_id": problem["task_id"],
                "passed": success,
            })
        
        return BenchmarkResult(
            benchmark="HumanEval",
            total=len(problems),
            passed=passed,
            pass_rate=passed / len(problems),
            details=details,
        )


class DS1000Runner:
    """
    Run DS-1000 benchmark (PyTorch subset).
    
    DS-1000 tests data science code generation including:
    - NumPy, Pandas, Matplotlib, Sklearn, PyTorch, TensorFlow
    
    We focus on the PyTorch subset (~100 problems) which includes:
    - Tensor operations
    - Model building
    - Training loops
    """
    
    def __init__(self, model_path: str):
        self.device = get_best_device()
        
        print(f"Loading model for DS-1000: {model_path}")
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
    
    def _load_ds1000_pytorch(self) -> list[dict]:
        """Load DS-1000 PyTorch subset."""
        try:
            from datasets import load_dataset
            # DS-1000 is organized by library
            dataset = load_dataset("xlangai/DS-1000", split="test")
            
            # Filter to PyTorch only
            pytorch_problems = [
                p for p in dataset 
                if p.get("lib") == "Pytorch" or "torch" in str(p.get("prompt", "")).lower()
            ]
            return pytorch_problems
        except Exception as e:
            print(f"⚠️  Could not load DS-1000: {e}")
            print("   Install with: pip install datasets")
            return []
    
    def _run_test(self, code: str, test_code: str) -> bool:
        """Run DS-1000 test case."""
        full_code = "import torch\nimport torch.nn as nn\n\n" + code + "\n\n" + test_code
        
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                f.flush()
                
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    timeout=30,
                    text=True,
                )
                
                os.unlink(f.name)
                return result.returncode == 0
        except Exception:
            return False
    
    def evaluate(
        self,
        max_samples: int = None,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> BenchmarkResult:
        """
        Run DS-1000 PyTorch subset.
        
        Args:
            max_samples: Limit number of problems
            temperature: Lower is more deterministic
            max_new_tokens: Max tokens to generate
            
        Returns:
            BenchmarkResult with pass rate
        """
        problems = self._load_ds1000_pytorch()
        if not problems:
            return BenchmarkResult("DS-1000-PyTorch", 0, 0, 0.0)
        
        if max_samples:
            problems = problems[:max_samples]
        
        passed = 0
        details = []
        
        print(f"\nRunning DS-1000 PyTorch on {len(problems)} problems...")
        
        for i, problem in enumerate(tqdm(problems, desc="DS-1000")):
            prompt = problem.get("prompt", "")
            test_code = problem.get("test", problem.get("test_code", ""))
            
            # Generate
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = generated[len(prompt):]  # Remove prompt
            
            # Test
            success = self._run_test(code, test_code) if test_code else False
            if success:
                passed += 1
            
            details.append({
                "problem_id": i,
                "passed": success,
            })
        
        return BenchmarkResult(
            benchmark="DS-1000-PyTorch",
            total=len(problems),
            passed=passed,
            pass_rate=passed / len(problems) if problems else 0.0,
            details=details,
        )

