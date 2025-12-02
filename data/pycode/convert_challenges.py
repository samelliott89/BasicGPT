#!/usr/bin/env python3
"""
Convert between challenges.jsonl and challenges.md formats.

Usage:
    # JSONL → Markdown (readable)
    python convert_challenges.py --to-md
    
    # Markdown → JSONL (for training)
    python convert_challenges.py --to-jsonl
"""

import argparse
import json
import re
from pathlib import Path

JSONL_PATH = Path(__file__).parent / "challenges.jsonl"
MD_PATH = Path(__file__).parent / "challenges.md"


def jsonl_to_markdown():
    """Convert challenges.jsonl to readable markdown."""
    with open(JSONL_PATH) as f:
        challenges = [json.loads(line) for line in f if line.strip()]
    
    output = "# Code Generation Challenges\n\n"
    output += f"Total: {len(challenges)} challenges\n\n"
    output += "---\n\n"
    
    for i, c in enumerate(challenges, 1):
        output += f"## Challenge {i}: {c['category'].upper()} ({c['difficulty']})\n\n"
        output += f"**Prompt:** {c['prompt']}\n\n"
        output += f"```python\n{c['code']}\n```\n\n"
        if c.get("test_code"):
            output += f"**Test:**\n```python\n{c['test_code']}\n```\n\n"
        output += "---\n\n"
    
    with open(MD_PATH, "w") as f:
        f.write(output)
    
    print(f"✓ Created {MD_PATH}")
    print(f"  {len(challenges)} challenges converted to markdown")


def markdown_to_jsonl():
    """Convert challenges.md back to JSONL for training."""
    with open(MD_PATH) as f:
        content = f.read()
    
    # Parse challenges
    pattern = r"## Challenge \d+: (\w+) \((\w+)\)\n\n\*\*Prompt:\*\* (.+?)\n\n```python\n(.+?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    
    challenges = []
    for category, difficulty, prompt, code in matches:
        challenges.append({
            "prompt": prompt.strip(),
            "code": code.strip(),
            "test_code": "",
            "difficulty": difficulty.lower(),
            "category": category.lower(),
        })
    
    with open(JSONL_PATH, "w") as f:
        for c in challenges:
            f.write(json.dumps(c) + "\n")
    
    print(f"✓ Created {JSONL_PATH}")
    print(f"  {len(challenges)} challenges converted to JSONL")


def main():
    parser = argparse.ArgumentParser(description="Convert challenge formats")
    parser.add_argument("--to-md", action="store_true", help="JSONL → Markdown")
    parser.add_argument("--to-jsonl", action="store_true", help="Markdown → JSONL")
    
    args = parser.parse_args()
    
    if args.to_md:
        jsonl_to_markdown()
    elif args.to_jsonl:
        markdown_to_jsonl()
    else:
        print("Usage:")
        print("  python convert_challenges.py --to-md      # JSONL → Markdown")
        print("  python convert_challenges.py --to-jsonl   # Markdown → JSONL")


if __name__ == "__main__":
    main()

