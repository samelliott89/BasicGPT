#!/usr/bin/env python3
"""
Merge multiple JSONL challenge files into one.

Usage:
    python scripts/merge_challenges.py \
        data/pycode/challenges.jsonl \
        data/pycode/timm_layers.jsonl \
        --output data/pycode/all_challenges.jsonl
"""

import argparse
import json
from pathlib import Path


def merge_files(input_files: list[str], output_file: str, dedupe: bool = True):
    """Merge multiple JSONL files."""
    all_examples = []
    seen_codes = set()
    
    for input_file in input_files:
        path = Path(input_file)
        if not path.exists():
            print(f"⚠️  File not found: {input_file}")
            continue
        
        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                    
                example = json.loads(line)
                
                # Dedupe by code content
                if dedupe:
                    code_hash = hash(example.get("code", ""))
                    if code_hash in seen_codes:
                        continue
                    seen_codes.add(code_hash)
                
                all_examples.append(example)
                count += 1
        
        print(f"  {path.name}: {count} examples")
    
    # Save merged file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    return len(all_examples)


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL challenge files")
    parser.add_argument(
        "files",
        nargs="+",
        help="Input JSONL files to merge",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output merged JSONL file",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Don't remove duplicate code blocks",
    )
    
    args = parser.parse_args()
    
    print("Merging challenge files:")
    total = merge_files(args.files, args.output, dedupe=not args.no_dedupe)
    print(f"\n✓ Merged {total} examples into {args.output}")


if __name__ == "__main__":
    main()

