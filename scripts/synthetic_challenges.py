#!/usr/bin/env python3
"""
Synthetic Challenge Generator

Takes real code from GitHub and generates prompts using a language model.
The key insight: real code is ground truth, we only synthesize the prompt.

Pipeline:
    1. Load real code (from scraped repos)
    2. Ask a model to generate a prompt that would produce this code
    3. Validate the (prompt, code) pair through execution
    4. Save validated pairs as training data

Usage:
    python scripts/synthetic_challenges.py \
        --input data/pycode/scraped.jsonl \
        --output data/pycode/synthetic.jsonl \
        --model deepseek-ai/deepseek-coder-1.3b-instruct
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.rl.code_rewards import CodeExecutor, compute_reward, reward_breakdown
from data.pycode.types import Source
from utils.device import get_best_device


# =============================================================================
# Prompt Generation
# =============================================================================

PROMPT_GENERATION_TEMPLATE = """Given this PyTorch code, write ONE sentence describing what to build.

Code:
{code}

Write a single sentence prompt (no code, no markdown, no explanation):"""


def generate_prompt_for_code(
    code: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 100,
) -> str:
    """
    Use a language model to generate a prompt for given code.
    
    Args:
        code: The nn.Module implementation
        model: Language model for generation
        tokenizer: Tokenizer
        device: Device to run on
        max_new_tokens: Max tokens to generate
        
    Returns:
        Generated prompt string
    """
    # Format the meta-prompt
    input_text = PROMPT_GENERATION_TEMPLATE.format(code=code)
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract just the generated part
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the prompt (after our template)
    if "Write only the prompt" in full_output:
        prompt = full_output.split("Write only the prompt, nothing else:")[-1].strip()
    else:
        prompt = full_output[len(input_text):].strip()
    
    # Clean up common LLM preambles
    prompt = _clean_llm_response(prompt)
    
    # Take first meaningful line if multiple
    if "\n" in prompt:
        for line in prompt.split("\n"):
            line = line.strip()
            if line and len(line) > 15:  # Skip short filler lines
                prompt = line
                break
    
    return prompt


def _clean_llm_response(text: str) -> str:
    """Remove common LLM response patterns/preambles."""
    import re
    
    text = text.strip()
    
    # Remove markdown code fences entirely
    text = re.sub(r'```\w*\n?', '', text)
    text = re.sub(r'``\w*\n?', '', text)
    
    # Patterns to strip from the start (case-insensitive check, exact removal)
    prefixes = [
        # Acknowledgments
        "ok, ", "ok. ", "okay, ", "okay. ",
        "sure, ", "sure. ", "sure! ",
        "certainly, ", "certainly. ",
        "of course, ", "of course. ",
        "absolutely, ", "absolutely. ",
        # Common lead-ins
        "here's a prompt: ", "here's the prompt: ",
        "here is a prompt: ", "here is the prompt: ",
        "here's a prompt ", "here's the prompt ",
        "here is a prompt ", "here is the prompt ",
        "sure, here's a prompt: ", "sure, here's the prompt: ",
        "sure, here is a prompt: ", "sure, here is the prompt: ",
        "here you go: ", "here you go. ",
        # Labels
        "prompt: ", "the prompt: ", "the prompt is: ",
        "output: ", "response: ", "answer: ",
        "explanation: ", "- ",
    ]
    
    # Keep stripping until nothing matches
    changed = True
    while changed:
        changed = False
        text = text.strip()
        text_lower = text.lower()
        
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
                break
    
    # Strip quotes
    text = text.strip('"\'')
    
    # Remove markdown formatting
    if text.startswith("**") and "**" in text[2:]:
        text = text.replace("**", "")
    if text.startswith("`") and text.endswith("`"):
        text = text[1:-1]
    
    return text.strip()


# =============================================================================
# Validation through Execution
# =============================================================================

def validate_code(code: str, executor: CodeExecutor, test_code: str = "") -> dict:
    """
    Validate code through execution.
    
    Routes to appropriate validation based on code type.
    
    Returns:
        Dict with validation results and reward
    """
    result = executor.execute(code, test_code)
    reward = compute_reward(result)
    breakdown = reward_breakdown(result)
    
    return {
        "code_type": result.code_type.value,
        "parses": result.parses,
        "runs": result.runs,
        "reward": reward,
        "breakdown": breakdown,
        "error": result.error_msg,
        # Type-specific fields
        "instantiates": result.instantiates,
        "forward_works": result.forward_works,
        "backward_works": result.backward_works,
        "trains_stable": result.trains_stable,
        "callable": result.callable,
        "returns_value": result.returns_value,
        "tests_pass": result.tests_pass,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

class SyntheticChallengeGenerator:
    """
    Generate synthetic (prompt, code) pairs from real code.
    
    The code is REAL (from GitHub), only the prompt is synthetic.
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
        device: torch.device = None,
    ):
        self.device = device or get_best_device()
        
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Code executor for validation
        self.executor = CodeExecutor(timeout=10.0)
        
        print(f"✓ Ready on {self.device}")
    
    def process_single(
        self, code: str, original_prompt: str = None, test_code: str = ""
    ) -> dict | None:
        """
        Process a single code sample.
        
        Args:
            code: The Python code (nn.Module, function, class, or script)
            original_prompt: Optional original prompt (for comparison)
            test_code: Optional test code to validate behavior
            
        Returns:
            Dict with prompt, code, validation results, or None if invalid
        """
        # Step 1: Validate the code actually works
        validation = validate_code(code, self.executor, test_code)
        
        if not validation["parses"]:
            return None  # Skip code that doesn't even parse
        
        # Step 2: Generate a prompt for this code
        generated_prompt = generate_prompt_for_code(
            code=code,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        
        if not generated_prompt or len(generated_prompt) < 10:
            return None  # Skip if prompt generation failed
        
        return {
            "prompt": generated_prompt,
            "code": code,
            "original_prompt": original_prompt,
            "validation": validation,
        }
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        max_samples: int = None,
        min_reward: float = 0.0,  # Minimum reward to include
    ) -> int:
        """
        Process a JSONL file of code samples.
        
        Args:
            input_path: Input JSONL with scraped code
            output_path: Output JSONL with synthetic challenges
            max_samples: Limit samples to process
            min_reward: Minimum execution reward to include
            
        Returns:
            Number of valid samples saved
        """
        # Load input
        samples = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"Processing {len(samples)} code samples...")
        
        # Process each sample
        valid_samples = []
        
        for sample in tqdm(samples, desc="Generating prompts"):
            code = sample.get("code", "")
            original_prompt = sample.get("prompt", "")
            test_code = sample.get("test_code", "")
            
            result = self.process_single(code, original_prompt, test_code)
            
            if result and result["validation"]["reward"] >= min_reward:
                # Create training example
                example = {
                    "prompt": result["prompt"],
                    "code": result["code"],
                    "test_code": "",
                    "difficulty": "medium",
                    "category": sample.get("category", "other"),
                    "source": Source.SYNTHETIC.value,
                    "metadata": {
                        "original_prompt": result["original_prompt"],
                        "original_source": sample.get("source", "unknown"),
                        "validation": result["validation"],
                    }
                }
                valid_samples.append(example)
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for example in valid_samples:
                f.write(json.dumps(example) + "\n")
        
        print(f"\n✓ Saved {len(valid_samples)} synthetic challenges to {output_path}")
        
        # Stats
        if valid_samples:
            rewards = [s["metadata"]["validation"]["reward"] for s in valid_samples]
            print(f"  Avg reward: {sum(rewards)/len(rewards):.3f}")
            print(f"  Parse rate: {sum(1 for s in valid_samples if s['metadata']['validation']['parses'])/len(valid_samples):.0%}")
            
            # Code type breakdown
            from collections import Counter
            type_counts = Counter(s["metadata"]["validation"]["code_type"] for s in valid_samples)
            print(f"  Code types: {dict(type_counts)}")
        
        return len(valid_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic challenges from real code"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with scraped code",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for synthetic challenges",
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/deepseek-coder-1.3b-instruct",
        help="Model to use for prompt generation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples to process",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.1,
        help="Minimum execution reward to include (default: 0.1 = must parse)",
    )
    
    args = parser.parse_args()
    
    # Suppress warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Synthetic Challenge Generator")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print()
    
    generator = SyntheticChallengeGenerator(model_name=args.model)
    generator.process_file(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
        min_reward=args.min_reward,
    )


if __name__ == "__main__":
    main()

