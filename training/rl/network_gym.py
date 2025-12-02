"""
CNN/Transformer Code Generation RL Environment

Reward signal from actual code execution, not just pattern matching.
"""

import ast
import os
import subprocess
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from utils.device import Device, get_best_device
from pathlib import Path


@dataclass
class ExecutionResult:
    parses: bool
    instantiates: bool
    forward_works: bool
    backward_works: bool
    trains_stable: bool  # No NaN after N steps
    error_msg: str | None = None

class CodeGenEnv(gym.Env):
    """
    RL environment for CNN/Transformer code generation.
    
    Observation: Tokenized prompt
    Action: Generated code (as token sequence)
    Reward: Based on execution success hierarchy
    """

    def __init__(
        self,
        challenges: list,
        tokenizer,
        max_tokens: int = 512,
        timeout: float = 10.0,
        device: Device = Device.CUDA,
        seed: int = 1337,
        torch_threads: int = 1,
    ):
        super().__init__()
        self.challenges = challenges
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.device = get_best_device()
        self.seed = seed
        self.torch_threads = torch_threads

        # Track current challenge
        self.current_idx = 0
        self.current_prompt = None
        self.current_test = None

        # Spaces (simplified - real impl would be more complex)
        vocab_size = tokenizer.vocab_size
        self.observation_space = spaces.Box(
            low=0, high=vocab_size, shape=(max_tokens,), dtype=np.int64
        )
        self.action_space = spaces.Box(
            low=0, high=vocab_size, shape=(max_tokens,), dtype=np.int64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample a challenge
        self.current_idx = np.random.randint(len(self.challenges))
        challenge = self.challenges[self.current_idx]
        self.current_prompt = challenge['prompt']
        self.current_test = challenge['test_code']

        # Tokenize prompt as observation
        tokens = self.tokenizer.encode(
            self.current_prompt,
            max_length=self.max_tokens,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )[0]

        return tokens, {"prompt": self.current_prompt}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Action is generated code tokens.
        Returns (obs, reward, terminated, truncated, info)
        """
        # Decode action to code string
        generated_code = self.tokenizer.decode(action, skip_special_tokens=True)

        # Execute and get hierarchical reward
        result = self._execute_code(generated_code)
        reward = self._compute_reward(result)

        # Episode ends after one generation attempt
        terminated = True
        truncated = False

        info = {
            "execution_result": result,
            "generated_code": generated_code,
            "reward_breakdown": self._reward_breakdown(result)
        }

        # Next observation (not used since terminated)
        next_obs = np.zeros(self.max_tokens, dtype=np.int64)

        return next_obs, reward, terminated, truncated, info

    def _execute_code(self, code: str) -> ExecutionResult:
        """Sandboxed execution with hierarchical checks."""

        result = ExecutionResult(
            parses=False,
            instantiates=False,
            forward_works=False,
            backward_works=False,
            trains_stable=False
        )

        # Level 1: Does it parse?
        try:
            ast.parse(code)
            result.parses = True
        except SyntaxError as e:
            result.error_msg = f"SyntaxError: {e}"
            return result

        # Level 2+: Run in subprocess sandbox
        test_script = self._build_test_script(code)

        try:
            # Constrain threads and set deterministic hash seed
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(self.torch_threads)
            env["MKL_NUM_THREADS"] = str(self.torch_threads)
            env["PYTHONHASHSEED"] = str(self.seed)

            proc = subprocess.run(
                ["python", "-c", test_script],
                capture_output=True,
                timeout=self.timeout,
                text=True,
                env=env,
            )

            output = proc.stdout

            # Parse structured output from test script
            if "INSTANTIATE_OK" in output:
                result.instantiates = True
            if "FORWARD_OK" in output:
                result.forward_works = True
            if "BACKWARD_OK" in output:
                result.backward_works = True
            if "TRAIN_STABLE" in output:
                result.trains_stable = True

            if proc.returncode != 0:
                result.error_msg = proc.stderr[:500]

        except subprocess.TimeoutExpired:
            result.error_msg = "Timeout"
        except Exception as e:
            result.error_msg = str(e)

        return result

    def _build_test_script(self, code: str) -> str:
        """
        Build the test script by loading the template from data/pycode/test.py
        and injecting the generated code plus the challenge-specific test block.
        
        The template uses literal placeholders:
          - {code}                -> replaced with the generated model code
          - {self.current_test}   -> replaced with the test harness code for the current challenge
          - {seed}                -> integer for deterministic seeding
          - {torch_threads}       -> integer for torch.set_num_threads
          - {device}              -> string descriptor of device type (cuda/mps/cpu)
          - {timeout}             -> float seconds (informational; subprocess enforces)
        
        Optionally, you can add more placeholders and replace them similarly.
        """
        # Resolve project root: <repo_root>/training/rl/network_gym.py -> parents[2]
        project_root = Path(__file__).resolve().parents[2]
        template_path = project_root / "data" / "pycode" / "test.py"
        
        template = template_path.read_text()
        
        # Simple, explicit replacements (avoid str.format to prevent `{}` conflicts)
        script = template.replace("{code}", code)
        script = script.replace("{self.current_test}", self.current_test or "")
        script = script.replace("{seed}", str(self.seed))
        script = script.replace("{torch_threads}", str(self.torch_threads))
        script = script.replace("{device}", str(getattr(self.device, "type", "cpu")))
        script = script.replace("{timeout}", str(self.timeout))
        
        return script

    def _compute_reward(self, result: ExecutionResult) -> float:
        """
        Hierarchical reward - each level unlocks more reward.
        Shaped to encourage incremental progress.
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

    def _reward_breakdown(self, result: ExecutionResult) -> dict:
        return {
            "parses": 0.1 if result.parses else 0,
            "instantiates": 0.2 if result.instantiates else 0,
            "forward": 0.3 if result.forward_works else 0,
            "backward": 0.2 if result.backward_works else 0,
            "stable": 0.2 if result.trains_stable else 0,
        }


# =============================================================================
# PPO Training Loop (simplified)
# =============================================================================

def train_with_ppo(
    policy_model,      # The LLM being trained
    ref_model,         # Frozen reference for KL penalty
    env: CodeGenEnv,
    tokenizer,
    num_episodes: int = 10000,
    kl_coef: float = 0.1,
):
    """
    Simplified PPO loop for code generation.
    
    In practice you'd use TRL's PPOTrainer or similar.
    """
    import torch.nn.functional as F

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

    for episode in range(num_episodes):
        obs, info = env.reset()
        prompt = info['prompt']

        # Generate from policy
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(policy_model.device)

        with torch.no_grad():
            # Sample from policy
            generated = policy_model.generate(
                prompt_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = generated.sequences[0]

        # Get reward from environment
        action = generated_ids.cpu().numpy()
        action = np.pad(action, (0, env.max_tokens - len(action)))[:env.max_tokens]
        _, reward, _, _, info = env.step(action)

        # Compute log probs under policy and reference
        with torch.no_grad():
            ref_logits = ref_model(generated_ids.unsqueeze(0)).logits
        policy_logits = policy_model(generated_ids.unsqueeze(0)).logits

        # KL divergence penalty
        kl = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction='batchmean'
        )

        # PPO objective (simplified - real impl needs advantages, clipping, etc.)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, generated_ids.unsqueeze(0).unsqueeze(-1))

        loss = -(reward - kl_coef * kl) * selected_log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: reward={reward:.3f}, kl={kl:.3f}")
            print(f"  Breakdown: {info['reward_breakdown']}")
