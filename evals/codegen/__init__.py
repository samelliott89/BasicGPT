"""
Code Generation Evaluation Suite

Three types of evaluation:
1. Execution-based: Does the generated code actually run?
2. Standard benchmarks: HumanEval, DS-1000 (PyTorch subset)
3. Held-out test set: Custom challenges not seen during training
"""

from evals.codegen.execution_eval import ExecutionEvaluator
from evals.codegen.benchmarks import HumanEvalRunner, DS1000Runner
from evals.codegen.held_out import HeldOutEvaluator
from evals.codegen.runner import CodeGenEvalRunner

__all__ = [
    "ExecutionEvaluator",
    "HumanEvalRunner",
    "DS1000Runner",
    "HeldOutEvaluator",
    "CodeGenEvalRunner",
]

