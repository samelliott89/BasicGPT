"""
Code Generation RL Trainer

Train with execution feedback using CodeGenEnv.
Rewards come from actual code execution, not pattern matching.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.codegen.config import CodeGenRLConfig, EXAMPLE_CHALLENGES
from training.rl.network_gym import CodeGenEnv, ExecutionResult
from utils.device import get_best_device


class CodeGenRLTrainer:
    """
    RL trainer for code generation with execution feedback.
    
    Uses simplified PPO-style training:
    1. Generate code from prompt
    2. Execute in sandbox (CodeGenEnv)
    3. Get hierarchical reward (parse → instantiate → forward → backward → stable)
    4. Update policy with KL-constrained gradient
    
    Example:
        config = CodeGenRLConfig(
            model_name="./checkpoints/codegen_sft/best",
            challenges_path="./data/pycode/challenges.jsonl",
        )
        trainer = CodeGenRLTrainer(config)
        trainer.train()
    """
    
    def __init__(self, config: CodeGenRLConfig):
        self.config = config
        self.device = get_best_device()
        
        # Load policy model (the one we're training)
        # Use float32 for stable gradient updates
        print(f"Loading policy model: {config.model_name}")
        self.policy = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,  # float32 for training stability
            trust_remote_code=True,
        ).to(self.device)
        
        # Load reference model (frozen, for KL penalty)
        print(f"Loading reference model: {config.ref_model_name}")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.ref_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.ref_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load challenges
        self.challenges = self._load_challenges()
        
        # Create environment
        # Note: env max_tokens is for the full sequence (prompt + response)
        # We use max_new_tokens for generation, env uses it for action space
        self.env = CodeGenEnv(
            challenges=self.challenges,
            tokenizer=self.tokenizer,
            max_tokens=config.max_prompt_length + config.max_new_tokens,
            timeout=config.execution_timeout,
            seed=config.seed,
            torch_threads=config.torch_threads,
        )
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics = {
            "rewards": [],
            "kl_divs": [],
            "parse_rates": [],
            "forward_rates": [],
            "stable_rates": [],
        }
        
        print(f"✓ Ready with {len(self.challenges)} challenges")
    
    def _load_challenges(self) -> list[dict]:
        """Load challenges from JSONL or use examples."""
        path = Path(self.config.challenges_path)
        
        if path.exists():
            print(f"Loading challenges from: {path}")
            challenges = []
            with open(path) as f:
                for line in f:
                    challenges.append(json.loads(line))
            return challenges
        else:
            print("⚠️  Using example challenges")
            return EXAMPLE_CHALLENGES
    
    def train(self):
        """Run RL training loop."""
        print("\n" + "=" * 60)
        print("Starting Code Generation RL Training")
        print("=" * 60)
        
        print(f"\nConfiguration:")
        print(f"  Policy: {self.config.model_name}")
        print(f"  Reference: {self.config.ref_model_name}")
        print(f"  Episodes: {self.config.num_episodes}")
        print(f"  KL coefficient: {self.config.kl_coef}")
        print(f"  Execution timeout: {self.config.execution_timeout}s")
        print()
        
        optimizer = AdamW(self.policy.parameters(), lr=self.config.learning_rate)
        
        # Rolling averages for logging
        recent_rewards = []
        recent_parses = []
        recent_forwards = []
        recent_stables = []
        
        progress_bar = tqdm(range(self.config.num_episodes), desc="RL Training")
        
        for episode in progress_bar:
            try:
                # Sample a challenge
                obs, info = self.env.reset()
                prompt = info["prompt"]
                
                # Format prompt for generation
                formatted_prompt = f"# Task: {prompt}\n\nimport torch\nimport torch.nn as nn\n\nclass "
                
                # Tokenize prompt (respecting max length to ensure full prompt fits)
                input_ids = self.tokenizer(
                    formatted_prompt,
                    max_length=self.config.max_prompt_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(self.device)
                
                # Set eval mode for generation, then back to train
                self.policy.eval()
                with torch.no_grad():
                    generated = self.policy.generate(
                        input_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                self.policy.train()
                
                generated_ids = generated[0]  # Direct tensor, not dict
                generated_code = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract just the class definition
                if "class " in generated_code:
                    code_start = generated_code.find("class ")
                    generated_code = generated_code[code_start:]
                
                # Execute and get reward
                result = self.env._execute_code(generated_code)
                reward = self.env._compute_reward(result)
                
                # Only update if we have new tokens
                prompt_len = input_ids.shape[1]
                new_tokens = generated_ids[prompt_len:]
                
                if len(new_tokens) < 2:
                    # Skip if generation was too short
                    continue
                
                # Get logits for generated tokens only
                gen_input = generated_ids.unsqueeze(0)
                
                with torch.no_grad():
                    ref_outputs = self.ref_model(gen_input)
                    ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
                
                policy_outputs = self.policy(gen_input)
                policy_logits = policy_outputs.logits[:, prompt_len-1:-1, :]
                
                # Cast to float32 for numerical stability in loss computation
                policy_logits_f32 = policy_logits.float()
                ref_logits_f32 = ref_logits.float()
                
                # Compute per-token log probs in float32
                policy_log_probs = F.log_softmax(policy_logits_f32, dim=-1)
                ref_log_probs = F.log_softmax(ref_logits_f32, dim=-1)
                
                # Gather log probs for actual tokens
                token_indices = new_tokens.unsqueeze(0).unsqueeze(-1)
                policy_token_lp = policy_log_probs.gather(-1, token_indices).squeeze(-1)
                ref_token_lp = ref_log_probs.gather(-1, token_indices).squeeze(-1)
                
                # Approximate KL from log prob difference
                kl_per_token = (policy_token_lp - ref_token_lp).detach()
                kl_div = kl_per_token.mean().abs()
                
                # Only update if we got some reward (at least parsed)
                if reward > 0 and not self.config.eval_only:
                    # Advantage: scale reward, penalize KL divergence
                    # Clip advantage to prevent too large updates
                    advantage = reward - self.config.kl_coef * kl_div.item()
                    advantage = max(-0.5, min(0.5, advantage))  # Clip to [-0.5, 0.5]
                    
                    # Policy gradient loss (REINFORCE)
                    loss = -(advantage * policy_token_lp.mean())
                    
                    # Skip if NaN
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Check for NaN gradients
                        has_nan = False
                        for p in self.policy.parameters():
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                has_nan = True
                                break
                        
                        if not has_nan:
                            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                            optimizer.step()
                        else:
                            optimizer.zero_grad()
                
                # Clear CUDA cache periodically
                if episode % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Track metrics
                recent_rewards.append(reward)
                recent_parses.append(1.0 if result.parses else 0.0)
                recent_forwards.append(1.0 if result.forward_works else 0.0)
                recent_stables.append(1.0 if result.trains_stable else 0.0)
                
                # Keep rolling window
                window = 100
                if len(recent_rewards) > window:
                    recent_rewards = recent_rewards[-window:]
                    recent_parses = recent_parses[-window:]
                    recent_forwards = recent_forwards[-window:]
                    recent_stables = recent_stables[-window:]
                
                # Update progress bar
                if episode % 10 == 0 and recent_rewards:
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    parse_rate = sum(recent_parses) / len(recent_parses)
                    forward_rate = sum(recent_forwards) / len(recent_forwards)
                    stable_rate = sum(recent_stables) / len(recent_stables)
                    
                    progress_bar.set_postfix({
                        "reward": f"{avg_reward:.3f}",
                        "parse": f"{parse_rate:.0%}",
                        "fwd": f"{forward_rate:.0%}",
                        "stable": f"{stable_rate:.0%}",
                        "kl": f"{kl_div.item():.3f}",
                    })
                
                # Logging
                if (episode + 1) % self.config.logging_steps == 0:
                    self._log_episode(episode, result, reward, kl_div.item(), generated_code)
                
                # Save checkpoint
                if (episode + 1) % self.config.save_steps == 0:
                    self._save_checkpoint(episode + 1)
                    
            except Exception as e:
                # Log error but continue training
                if episode < 5:
                    print(f"\n⚠️  Episode {episode} failed: {str(e)[:100]}")
                continue
        
        # Final save
        self._save_checkpoint(self.config.num_episodes, name="final")
        
        print("\n" + "=" * 60)
        print("RL Training Complete!")
        print("=" * 60)
        
        self._print_summary()
    
    def _log_episode(
        self,
        episode: int,
        result: ExecutionResult,
        reward: float,
        kl_div: float,
        code: str,
    ):
        """Log episode details."""
        self.metrics["rewards"].append(reward)
        self.metrics["kl_divs"].append(kl_div)
        self.metrics["parse_rates"].append(1.0 if result.parses else 0.0)
        self.metrics["forward_rates"].append(1.0 if result.forward_works else 0.0)
        self.metrics["stable_rates"].append(1.0 if result.trains_stable else 0.0)
    
    def _save_checkpoint(self, episode: int, name: str = None):
        """Save model checkpoint."""
        checkpoint_name = name or f"episode_{episode}"
        checkpoint_path = self.output_dir / checkpoint_name
        
        print(f"\n💾 Saving: {checkpoint_name}")
        
        self.policy.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        torch.save({
            "episode": episode,
            "metrics": self.metrics,
            "config": self.config.__dict__,
        }, checkpoint_path / "training_state.pt")
    
    def _print_summary(self):
        """Print training summary."""
        if not self.metrics["rewards"]:
            return
        
        print("\nTraining Summary:")
        print("-" * 40)
        
        # Recent vs early comparison
        n = len(self.metrics["rewards"])
        early = self.metrics["rewards"][:n//4] if n > 4 else self.metrics["rewards"]
        late = self.metrics["rewards"][-n//4:] if n > 4 else self.metrics["rewards"]
        
        print(f"  Early avg reward: {sum(early)/len(early):.3f}")
        print(f"  Late avg reward:  {sum(late)/len(late):.3f}")
        
        late_parse = self.metrics["parse_rates"][-n//4:] if n > 4 else self.metrics["parse_rates"]
        late_fwd = self.metrics["forward_rates"][-n//4:] if n > 4 else self.metrics["forward_rates"]
        late_stable = self.metrics["stable_rates"][-n//4:] if n > 4 else self.metrics["stable_rates"]
        
        print(f"\nFinal rates:")
        print(f"  Parse:   {sum(late_parse)/len(late_parse):.0%}")
        print(f"  Forward: {sum(late_fwd)/len(late_fwd):.0%}")
        print(f"  Stable:  {sum(late_stable)/len(late_stable):.0%}")
    
    def generate(self, prompt: str) -> str:
        """Generate code from prompt using trained policy."""
        self.policy.eval()
        
        formatted = f"# Task: {prompt}\n\nimport torch\nimport torch.nn as nn\n\nclass "
        
        input_ids = self.tokenizer(
            formatted,
            max_length=self.config.max_prompt_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.policy.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "class " in generated:
            return generated[generated.find("class "):]
        return generated


if __name__ == "__main__":
    print("Code Generation RL Trainer")
    print("=" * 40)
    print()
    print("Prerequisites:")
    print("  1. Run SFT first to get a good starting point")
    print("  2. Have challenges.jsonl with test cases")
    print()
    print("Usage:")
    print("  from training.codegen import CodeGenRLTrainer, CodeGenRLConfig")
    print()
    print("  config = CodeGenRLConfig(")
    print('      model_name="./checkpoints/codegen_sft/best",')
    print('      challenges_path="./data/pycode/challenges.jsonl",')
    print("  )")
    print("  trainer = CodeGenRLTrainer(config)")
    print("  trainer.train()")

