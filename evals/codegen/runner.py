"""
Unified Code Generation Evaluation Runner

Runs all evaluations:
1. Execution-based (CodeGenEnv)
2. HumanEval (standard benchmark)
3. DS-1000 PyTorch (standard benchmark)
4. Held-out test set (custom)
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from evals.codegen.execution_eval import ExecutionEvaluator, ExecutionMetrics
from evals.codegen.benchmarks import HumanEvalRunner, DS1000Runner, BenchmarkResult
from evals.codegen.held_out import HeldOutEvaluator, HeldOutMetrics


@dataclass
class FullEvalResults:
    """Complete evaluation results."""
    model_path: str
    timestamp: str
    execution: ExecutionMetrics | None = None
    humaneval: BenchmarkResult | None = None
    ds1000: BenchmarkResult | None = None
    held_out: HeldOutMetrics | None = None
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CODE GENERATION EVALUATION RESULTS",
            "=" * 60,
            f"Model: {self.model_path}",
            f"Time: {self.timestamp}",
            "",
        ]
        
        if self.execution:
            lines.extend([
                "EXECUTION-BASED EVAL",
                "-" * 40,
                str(self.execution),
                "",
            ])
        
        if self.humaneval:
            lines.extend([
                "HUMANEVAL",
                "-" * 40,
                str(self.humaneval),
                "",
            ])
        
        if self.ds1000:
            lines.extend([
                "DS-1000 (PYTORCH)",
                "-" * 40,
                str(self.ds1000),
                "",
            ])
        
        if self.held_out:
            lines.extend([
                "HELD-OUT TEST SET",
                "-" * 40,
                str(self.held_out),
                "",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "execution": asdict(self.execution) if self.execution else None,
            "humaneval": asdict(self.humaneval) if self.humaneval else None,
            "ds1000": asdict(self.ds1000) if self.ds1000 else None,
            "held_out": asdict(self.held_out) if self.held_out else None,
        }


class CodeGenEvalRunner:
    """
    Run comprehensive code generation evaluation.
    
    Example:
        runner = CodeGenEvalRunner("./checkpoints/codegen_sft/best")
        results = runner.run_all()
        print(results.summary())
    """
    
    def __init__(
        self,
        model_path: str,
        challenges_path: str = "./data/pycode/challenges.jsonl",
        test_path: str = "./data/pycode/test_challenges.jsonl",
        output_dir: str = "./evals/results",
    ):
        self.model_path = model_path
        self.challenges_path = challenges_path
        self.test_path = test_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_execution(self, max_samples: int = None) -> ExecutionMetrics:
        """Run execution-based evaluation."""
        print("\n" + "=" * 60)
        print("EXECUTION-BASED EVALUATION")
        print("=" * 60)
        
        evaluator = ExecutionEvaluator(
            model_path=self.model_path,
            challenges_path=self.challenges_path,
        )
        return evaluator.evaluate(max_samples=max_samples)
    
    def run_humaneval(self, max_samples: int = None) -> BenchmarkResult:
        """Run HumanEval benchmark."""
        print("\n" + "=" * 60)
        print("HUMANEVAL BENCHMARK")
        print("=" * 60)
        
        runner = HumanEvalRunner(self.model_path)
        return runner.evaluate(max_samples=max_samples)
    
    def run_ds1000(self, max_samples: int = None) -> BenchmarkResult:
        """Run DS-1000 PyTorch subset."""
        print("\n" + "=" * 60)
        print("DS-1000 PYTORCH BENCHMARK")
        print("=" * 60)
        
        runner = DS1000Runner(self.model_path)
        return runner.evaluate(max_samples=max_samples)
    
    def run_held_out(self) -> HeldOutMetrics:
        """Run held-out test set evaluation."""
        print("\n" + "=" * 60)
        print("HELD-OUT TEST SET EVALUATION")
        print("=" * 60)
        
        evaluator = HeldOutEvaluator(
            model_path=self.model_path,
            test_path=self.test_path,
        )
        return evaluator.evaluate()
    
    def run_all(
        self,
        skip_humaneval: bool = False,
        skip_ds1000: bool = False,
        skip_held_out: bool = False,
        max_samples: int = None,
    ) -> FullEvalResults:
        """
        Run all evaluations.
        
        Args:
            skip_humaneval: Skip HumanEval (requires dataset download)
            skip_ds1000: Skip DS-1000 (requires dataset download)
            skip_held_out: Skip held-out (requires train/test split)
            max_samples: Limit samples per eval (for quick testing)
            
        Returns:
            FullEvalResults with all metrics
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results = FullEvalResults(
            model_path=self.model_path,
            timestamp=timestamp,
        )
        
        # Always run execution-based
        results.execution = self.run_execution(max_samples)
        
        # Optional benchmarks
        if not skip_humaneval:
            try:
                results.humaneval = self.run_humaneval(max_samples)
            except Exception as e:
                print(f"⚠️  HumanEval failed: {e}")
        
        if not skip_ds1000:
            try:
                results.ds1000 = self.run_ds1000(max_samples)
            except Exception as e:
                print(f"⚠️  DS-1000 failed: {e}")
        
        if not skip_held_out:
            try:
                results.held_out = self.run_held_out()
            except Exception as e:
                print(f"⚠️  Held-out eval failed: {e}")
        
        # Print summary
        print("\n" + results.summary())
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: FullEvalResults):
        """Save results to JSON file."""
        model_name = Path(self.model_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file = self.output_dir / f"eval_{model_name}_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {output_file}")

