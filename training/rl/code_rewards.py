"""
Code Type Classification and Reward Functions

Routes different types of Python code to appropriate validation and reward functions.
"""

import ast
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CodeType(Enum):
    """Types of Python code we can validate."""
    NN_MODULE = "nn_module"       # PyTorch nn.Module class
    FUNCTION = "function"         # Standalone function
    CLASS = "class"               # Generic class (not nn.Module)
    SCRIPT = "script"             # Executable script/statements
    UNKNOWN = "unknown"           # Couldn't classify


@dataclass
class ExecutionResult:
    """Results from code execution."""
    code_type: CodeType
    parses: bool
    runs: bool  # Generic "does it run"
    
    # NN_MODULE specific
    instantiates: bool = False
    forward_works: bool = False
    backward_works: bool = False
    trains_stable: bool = False
    
    # FUNCTION specific
    callable: bool = False
    returns_value: bool = False
    
    # Test results (if test code provided)
    tests_pass: bool = False
    
    error_msg: str | None = None
    output: str | None = None
    
    # Extra metadata
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Code Type Classification
# =============================================================================

def classify_code(code: str) -> CodeType:
    """
    Classify Python code into a type for routing to appropriate reward function.
    
    Uses AST parsing to determine code structure.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return CodeType.UNKNOWN
    
    # Collect top-level definitions
    classes = []
    functions = []
    statements = []
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(node)
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            # Expr could be docstrings, skip those
            statements.append(node)
    
    # Check for nn.Module class
    for cls in classes:
        if _is_nn_module(cls):
            return CodeType.NN_MODULE
    
    # Has classes but not nn.Module
    if classes:
        return CodeType.CLASS
    
    # Only functions defined
    if functions and not statements:
        return CodeType.FUNCTION
    
    # Has executable statements
    if statements or (not classes and not functions):
        return CodeType.SCRIPT
    
    return CodeType.FUNCTION if functions else CodeType.UNKNOWN


def _is_nn_module(cls: ast.ClassDef) -> bool:
    """Check if a class inherits from nn.Module."""
    for base in cls.bases:
        base_name = _get_base_name(base)
        if base_name in ("nn.Module", "Module", "torch.nn.Module"):
            return True
    return False


def _get_base_name(node: ast.expr) -> str:
    """Extract base class name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


# =============================================================================
# Reward Functions by Code Type
# =============================================================================

def compute_reward(result: ExecutionResult) -> float:
    """
    Compute reward based on code type and execution results.
    
    Routes to appropriate reward function based on code type.
    """
    match result.code_type:
        case CodeType.NN_MODULE:
            return _nn_module_reward(result)
        case CodeType.FUNCTION:
            return _function_reward(result)
        case CodeType.CLASS:
            return _class_reward(result)
        case CodeType.SCRIPT:
            return _script_reward(result)
        case CodeType.UNKNOWN:
            return 0.0


def _nn_module_reward(result: ExecutionResult) -> float:
    """
    Hierarchical reward for nn.Module code.
    
    Levels:
        0.1 - Parses
        0.2 - Instantiates  
        0.3 - Forward works
        0.2 - Backward works
        0.2 - Trains stable
    """
    reward = 0.0
    
    if result.parses:
        reward += 0.1
    if result.instantiates:
        reward += 0.2
    if result.forward_works:
        reward += 0.3
    if result.backward_works:
        reward += 0.2
    if result.trains_stable:
        reward += 0.2
    
    return reward


def _function_reward(result: ExecutionResult) -> float:
    """
    Reward for standalone functions.
    
    Levels:
        0.2 - Parses
        0.3 - Callable (can be invoked)
        0.3 - Returns value without error
        0.2 - Tests pass (if provided)
    """
    reward = 0.0
    
    if result.parses:
        reward += 0.2
    if result.callable:
        reward += 0.3
    if result.returns_value:
        reward += 0.3
    if result.tests_pass:
        reward += 0.2
    
    return reward


def _class_reward(result: ExecutionResult) -> float:
    """
    Reward for generic classes.
    
    Levels:
        0.2 - Parses
        0.4 - Instantiates
        0.2 - Runs without error
        0.2 - Tests pass (if provided)
    """
    reward = 0.0
    
    if result.parses:
        reward += 0.2
    if result.instantiates:
        reward += 0.4
    if result.runs:
        reward += 0.2
    if result.tests_pass:
        reward += 0.2
    
    return reward


def _script_reward(result: ExecutionResult) -> float:
    """
    Reward for executable scripts.
    
    Levels:
        0.3 - Parses
        0.5 - Runs without error
        0.2 - Produces expected output (if provided)
    """
    reward = 0.0
    
    if result.parses:
        reward += 0.3
    if result.runs:
        reward += 0.5
    if result.tests_pass:
        reward += 0.2
    
    return reward


def reward_breakdown(result: ExecutionResult) -> dict:
    """Get detailed breakdown of reward components."""
    match result.code_type:
        case CodeType.NN_MODULE:
            return {
                "type": "nn_module",
                "parses": 0.1 if result.parses else 0,
                "instantiates": 0.2 if result.instantiates else 0,
                "forward": 0.3 if result.forward_works else 0,
                "backward": 0.2 if result.backward_works else 0,
                "stable": 0.2 if result.trains_stable else 0,
            }
        case CodeType.FUNCTION:
            return {
                "type": "function",
                "parses": 0.2 if result.parses else 0,
                "callable": 0.3 if result.callable else 0,
                "returns": 0.3 if result.returns_value else 0,
                "tests": 0.2 if result.tests_pass else 0,
            }
        case CodeType.CLASS:
            return {
                "type": "class",
                "parses": 0.2 if result.parses else 0,
                "instantiates": 0.4 if result.instantiates else 0,
                "runs": 0.2 if result.runs else 0,
                "tests": 0.2 if result.tests_pass else 0,
            }
        case CodeType.SCRIPT:
            return {
                "type": "script",
                "parses": 0.3 if result.parses else 0,
                "runs": 0.5 if result.runs else 0,
                "tests": 0.2 if result.tests_pass else 0,
            }
        case _:
            return {"type": "unknown", "total": 0}


# =============================================================================
# Code Execution
# =============================================================================

class CodeExecutor:
    """
    Execute Python code in a sandboxed subprocess.
    
    Routes to appropriate execution logic based on code type.
    """
    
    def __init__(self, timeout: float = 10.0, torch_threads: int = 1):
        self.timeout = timeout
        self.torch_threads = torch_threads
    
    def execute(self, code: str, test_code: str = "") -> ExecutionResult:
        """
        Execute code and return results.
        
        Args:
            code: The Python code to execute
            test_code: Optional test code to validate behavior
            
        Returns:
            ExecutionResult with validation results
        """
        code_type = classify_code(code)
        
        match code_type:
            case CodeType.NN_MODULE:
                return self._execute_nn_module(code, test_code)
            case CodeType.FUNCTION:
                return self._execute_function(code, test_code)
            case CodeType.CLASS:
                return self._execute_class(code, test_code)
            case CodeType.SCRIPT:
                return self._execute_script(code, test_code)
            case CodeType.UNKNOWN:
                return ExecutionResult(
                    code_type=CodeType.UNKNOWN,
                    parses=False,
                    runs=False,
                    error_msg="Could not parse code",
                )
    
    def _execute_nn_module(self, code: str, test_code: str) -> ExecutionResult:
        """Execute and validate nn.Module code."""
        result = ExecutionResult(
            code_type=CodeType.NN_MODULE,
            parses=False,
            runs=False,
        )
        
        # Build test script
        script = self._build_nn_module_script(code, test_code)
        
        # Run in subprocess
        output, error, success = self._run_script(script)
        
        # Parse results from output
        if success and output:
            result.parses = "PARSE_OK" in output
            result.instantiates = "INSTANTIATE_OK" in output
            result.forward_works = "FORWARD_OK" in output
            result.backward_works = "BACKWARD_OK" in output
            result.trains_stable = "TRAIN_OK" in output
            result.tests_pass = "TEST_OK" in output if test_code else True
            result.runs = result.parses
        
        if error:
            result.error_msg = error[:500]  # Truncate long errors
        
        result.output = output
        return result
    
    def _execute_function(self, code: str, test_code: str) -> ExecutionResult:
        """Execute and validate function code."""
        result = ExecutionResult(
            code_type=CodeType.FUNCTION,
            parses=False,
            runs=False,
        )
        
        script = self._build_function_script(code, test_code)
        output, error, success = self._run_script(script)
        
        if success and output:
            result.parses = "PARSE_OK" in output
            result.callable = "CALLABLE_OK" in output
            result.returns_value = "RETURNS_OK" in output
            result.tests_pass = "TEST_OK" in output if test_code else True
            result.runs = result.parses
        
        if error:
            result.error_msg = error[:500]
        
        result.output = output
        return result
    
    def _execute_class(self, code: str, test_code: str) -> ExecutionResult:
        """Execute and validate generic class code."""
        result = ExecutionResult(
            code_type=CodeType.CLASS,
            parses=False,
            runs=False,
        )
        
        script = self._build_class_script(code, test_code)
        output, error, success = self._run_script(script)
        
        if success and output:
            result.parses = "PARSE_OK" in output
            result.instantiates = "INSTANTIATE_OK" in output
            result.runs = "RUN_OK" in output
            result.tests_pass = "TEST_OK" in output if test_code else True
        
        if error:
            result.error_msg = error[:500]
        
        result.output = output
        return result
    
    def _execute_script(self, code: str, test_code: str) -> ExecutionResult:
        """Execute script code."""
        result = ExecutionResult(
            code_type=CodeType.SCRIPT,
            parses=False,
            runs=False,
        )
        
        script = self._build_script_script(code, test_code)
        output, error, success = self._run_script(script)
        
        if success and output:
            result.parses = "PARSE_OK" in output
            result.runs = "RUN_OK" in output
            result.tests_pass = "TEST_OK" in output if test_code else True
        
        if error:
            result.error_msg = error[:500]
        
        result.output = output
        return result
    
    def _run_script(self, script: str) -> tuple[str, str, bool]:
        """Run a Python script in subprocess."""
        import os
        import sys
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Preserve PATH to keep virtual environment
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": str(self.torch_threads),
                "MKL_NUM_THREADS": str(self.torch_threads),
            })
            
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            
            return proc.stdout, proc.stderr, proc.returncode == 0
            
        except subprocess.TimeoutExpired:
            return "", "Timeout", False
        except Exception as e:
            return "", str(e), False
        finally:
            Path(script_path).unlink(missing_ok=True)
    
    # =========================================================================
    # Script Builders
    # =========================================================================
    
    def _build_nn_module_script(self, code: str, test_code: str) -> str:
        """Build test script for nn.Module."""
        # Escape code for embedding in triple-quoted string
        escaped_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        
        test_block = ""
        if test_code:
            test_block = f'''
# Step 6: Custom tests
try:
    {test_code}
    print("TEST_OK")
except Exception as e:
    print(f"TEST_FAIL: {{e}}")
'''
        
        return f'''
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from torch.nn import Module, ModuleList, Linear, LayerNorm, Dropout, Embedding
from torch.nn import Conv1d, Conv2d, BatchNorm1d, BatchNorm2d

# Common aliases
try:
    from einops import rearrange, repeat, reduce
except ImportError:
    pass

# Step 1: Parse
try:
    exec("""{escaped_code}""")
    print("PARSE_OK")
except Exception as e:
    print(f"PARSE_FAIL: {{e}}")
    sys.exit(0)

# Find the Module class
module_cls = None
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
        module_cls = obj
        break

if module_cls is None:
    print("NO_MODULE_FOUND")
    sys.exit(0)

# Step 2: Instantiate
try:
    model = module_cls()
    print("INSTANTIATE_OK")
except Exception as e:
    print(f"INSTANTIATE_FAIL: {{e}}")
    sys.exit(0)

# Step 3: Forward pass
try:
    x = torch.randn(2, 3, 32, 32)  # Common input shape
    try:
        out = model(x)
    except:
        x = torch.randn(2, 64)  # Try 1D input
        out = model(x)
    print("FORWARD_OK")
except Exception as e:
    print(f"FORWARD_FAIL: {{e}}")
    sys.exit(0)

# Step 4: Backward pass
try:
    loss = out.mean()
    loss.backward()
    print("BACKWARD_OK")
except Exception as e:
    print(f"BACKWARD_FAIL: {{e}}")
    sys.exit(0)

# Step 5: Training stability (3 steps)
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        x = torch.randn(2, 3, 32, 32)
        try:
            out = model(x)
        except:
            x = torch.randn(2, 64)
            out = model(x)
        loss = out.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN/Inf in loss")
        loss.backward()
        optimizer.step()
    print("TRAIN_OK")
except Exception as e:
    print(f"TRAIN_FAIL: {{e}}")
{test_block}
'''
    
    def _build_function_script(self, code: str, test_code: str) -> str:
        """Build test script for functions."""
        escaped_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        
        test_block = ""
        if test_code:
            test_block = f'''
# Step 4: Custom tests
try:
    {test_code}
    print("TEST_OK")
except Exception as e:
    print(f"TEST_FAIL: {{e}}")
'''
        
        return f'''
import sys

# Step 1: Parse
try:
    exec("""{escaped_code}""")
    print("PARSE_OK")
except Exception as e:
    print(f"PARSE_FAIL: {{e}}")
    sys.exit(0)

# Find the function
func = None
for name, obj in list(locals().items()):
    if callable(obj) and not name.startswith('_') and name not in ('exec', 'print', 'sys'):
        func = obj
        break

if func is None:
    print("NO_FUNCTION_FOUND")
    sys.exit(0)

# Step 2: Check callable
try:
    if callable(func):
        print("CALLABLE_OK")
    else:
        print("CALLABLE_FAIL")
        sys.exit(0)
except Exception as e:
    print(f"CALLABLE_FAIL: {{e}}")
    sys.exit(0)

# Step 3: Try to call it
try:
    result = func()
    print("RETURNS_OK")
except TypeError:
    # Needs arguments, try with common defaults
    try:
        result = func(1)
        print("RETURNS_OK")
    except:
        try:
            result = func("test")
            print("RETURNS_OK")
        except:
            print("RETURNS_FAIL: needs specific arguments")
except Exception as e:
    print(f"RETURNS_FAIL: {{e}}")
{test_block}
'''
    
    def _build_class_script(self, code: str, test_code: str) -> str:
        """Build test script for generic classes."""
        escaped_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        
        test_block = ""
        if test_code:
            test_block = f'''
# Step 4: Custom tests
try:
    {test_code}
    print("TEST_OK")
except Exception as e:
    print(f"TEST_FAIL: {{e}}")
'''
        
        return f'''
import sys

# Step 1: Parse
try:
    exec("""{escaped_code}""")
    print("PARSE_OK")
except Exception as e:
    print(f"PARSE_FAIL: {{e}}")
    sys.exit(0)

# Find the class
cls = None
for name, obj in list(locals().items()):
    if isinstance(obj, type) and not name.startswith('_'):
        cls = obj
        break

if cls is None:
    print("NO_CLASS_FOUND")
    sys.exit(0)

# Step 2: Instantiate
try:
    instance = cls()
    print("INSTANTIATE_OK")
except TypeError:
    # Needs arguments
    try:
        instance = cls(1)
        print("INSTANTIATE_OK")
    except:
        print("INSTANTIATE_FAIL: needs specific arguments")
        sys.exit(0)
except Exception as e:
    print(f"INSTANTIATE_FAIL: {{e}}")
    sys.exit(0)

# Step 3: Basic run check
try:
    str(instance)  # Should at least be representable
    print("RUN_OK")
except Exception as e:
    print(f"RUN_FAIL: {{e}}")
{test_block}
'''
    
    def _build_script_script(self, code: str, test_code: str) -> str:
        """Build test script for scripts."""
        escaped_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        
        test_block = ""
        if test_code:
            test_block = f'''
# Step 3: Custom tests
try:
    {test_code}
    print("TEST_OK")
except Exception as e:
    print(f"TEST_FAIL: {{e}}")
'''
        
        return f'''
import sys

# Step 1: Parse
try:
    compile("""{escaped_code}""", "<string>", "exec")
    print("PARSE_OK")
except Exception as e:
    print(f"PARSE_FAIL: {{e}}")
    sys.exit(0)

# Step 2: Execute
try:
    exec("""{escaped_code}""")
    print("RUN_OK")
except Exception as e:
    print(f"RUN_FAIL: {{e}}")
{test_block}
'''

