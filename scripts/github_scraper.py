#!/usr/bin/env python3
"""
GitHub Code Scraper

Download Python files from GitHub repos and extract nn.Module definitions
for training data.

Usage:
    # Download all nn.Module classes from timm
    python scripts/github_scraper.py huggingface/pytorch-image-models \
        --paths timm/models \
        --output data/pycode/timm_modules.jsonl

    # Download specific attention implementations
    python scripts/github_scraper.py huggingface/transformers \
        --paths src/transformers/models/bert/modeling_bert.py \
        --output data/pycode/bert_attention.jsonl
        
    # Multiple paths
    python scripts/github_scraper.py timm/timm \
        --paths timm/layers timm/models/resnet.py \
        --output data/pycode/timm_all.jsonl
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pycode.types import get_source


@dataclass
class ExtractedCode:
    """A single extracted code block."""
    name: str           # Class or function name
    code: str           # Full source code
    docstring: str      # Docstring if present
    file_path: str      # Original file path
    start_line: int     # Line number in original file
    code_type: str      # "class" or "function"
    parent_class: str   # Parent class if method, else ""
    

class CodeExtractor(ast.NodeVisitor):
    """Extract nn.Module classes from Python AST."""
    
    def __init__(self, source: str, file_path: str):
        self.source = source
        self.source_lines = source.split('\n')
        self.file_path = file_path
        self.extracted: list[ExtractedCode] = []
        self.current_class = None
    
    def get_source_segment(self, node) -> str:
        """Get the source code for a node."""
        start_line = node.lineno - 1
        end_line = node.end_lineno
        lines = self.source_lines[start_line:end_line]
        return '\n'.join(lines)
    
    def get_docstring(self, node) -> str:
        """Extract docstring from a node."""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value.strip()
        return ""
    
    def is_nn_module(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from nn.Module."""
        for base in node.bases:
            base_str = ast.unparse(base)
            if 'Module' in base_str or 'nn.' in base_str:
                return True
        return False
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions."""
        # Check if it's an nn.Module subclass
        if self.is_nn_module(node):
            code = self.get_source_segment(node)
            docstring = self.get_docstring(node)
            
            self.extracted.append(ExtractedCode(
                name=node.name,
                code=code,
                docstring=docstring,
                file_path=self.file_path,
                start_line=node.lineno,
                code_type="class",
                parent_class="",
            ))
        
        # Visit child nodes (for nested classes)
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions - only for traversing into nested classes."""
        # We only extract nn.Module classes, not standalone functions
        # (standalone functions often have dependencies that make them unusable)
        self.generic_visit(node)
    
    visit_AsyncFunctionDef = visit_FunctionDef


def parse_github_url(url: str) -> tuple[str, str | None]:
    """
    Parse various GitHub URL formats.
    
    Handles:
        - lucidrains/x-transformers
        - https://github.com/lucidrains/x-transformers
        - https://github.com/lucidrains/x-transformers/tree/main/x_transformers
        
    Returns:
        (repo, path) - repo is "owner/name", path is the folder/file path or None
    """
    url = url.rstrip('/')
    
    # Already in owner/repo format
    if not url.startswith('http'):
        return url, None
    
    # Parse GitHub URL
    # https://github.com/lucidrains/x-transformers/tree/main/x_transformers
    parts = url.replace('https://github.com/', '').split('/')
    
    if len(parts) >= 2:
        repo = f"{parts[0]}/{parts[1]}"
        
        # Check if there's a path (after /tree/branch/ or /blob/branch/)
        if len(parts) > 4 and parts[2] in ('tree', 'blob'):
            # parts[3] is branch, parts[4:] is path
            path = '/'.join(parts[4:])
            return repo, path
        
        return repo, None
    
    raise ValueError(f"Could not parse GitHub URL: {url}")


def clone_repo(repo: str, temp_dir: str) -> Path:
    """Clone a GitHub repo to temp directory."""
    url = f"https://github.com/{repo}.git"
    repo_name = repo.split('/')[-1]
    clone_path = Path(temp_dir) / repo_name
    
    print(f"Cloning {url}...")
    subprocess.run(
        ['git', 'clone', '--depth', '1', url, str(clone_path)],
        capture_output=True,
        check=True,
    )
    
    return clone_path


def find_python_files(repo_path: Path, paths: list[str] | None) -> list[Path]:
    """Find all Python files in specified paths."""
    python_files = []
    
    if paths:
        for path in paths:
            full_path = repo_path / path
            if full_path.is_file() and full_path.suffix == '.py':
                python_files.append(full_path)
            elif full_path.is_dir():
                python_files.extend(full_path.rglob('*.py'))
    else:
        # Search entire repo
        python_files = list(repo_path.rglob('*.py'))
    
    # Filter out test files and __init__.py
    python_files = [
        f for f in python_files 
        if not any(x in str(f) for x in ['test_', '_test.py', 'tests/', '__pycache__'])
        and f.name != '__init__.py'
    ]
    
    return python_files


def extract_from_file(file_path: Path, repo_path: Path) -> list[ExtractedCode]:
    """Extract code blocks from a Python file."""
    try:
        source = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return []
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    relative_path = str(file_path.relative_to(repo_path))
    extractor = CodeExtractor(source, relative_path)
    extractor.visit(tree)
    
    return extractor.extracted


def generate_prompt_from_code(code: ExtractedCode) -> str:
    """Generate a natural language prompt from extracted code."""
    name = code.name
    
    # Use docstring if available
    if code.docstring:
        # Take first line of docstring
        first_line = code.docstring.split('\n')[0].strip()
        if len(first_line) > 20:
            return first_line
    
    # Generate from class/function name
    # Convert CamelCase to spaces
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', words)
    words = words.replace('_', ' ').lower()
    
    if code.code_type == "class":
        return f"Create a {words} neural network module"
    else:
        return f"Create a {words} function"


def save_to_jsonl(
    extracted: list[ExtractedCode], 
    output_path: str, 
    repo: str,
    min_lines: int = 5
):
    """Save extracted code to JSONL format for training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get source from lookup table
    source_str = get_source(repo)
    
    saved = 0
    with open(output_path, 'w') as f:
        for code in extracted:
            # Skip very short code
            if code.code.count('\n') < min_lines:
                continue
            
            # Generate training example
            prompt = generate_prompt_from_code(code)
            
            example = {
                "prompt": prompt,
                "code": code.code,
                "test_code": "",  # Could add test generation later
                "difficulty": "medium",
                "category": categorize_code(code),
                "source": source_str,
                "metadata": {
                    "source_file": code.file_path,
                    "name": code.name,
                    "line": code.start_line,
                }
            }
            
            f.write(json.dumps(example) + '\n')
            saved += 1
    
    return saved, source_str


def categorize_code(code: ExtractedCode) -> str:
    """Categorize code based on name and content."""
    name_lower = code.name.lower()
    code_lower = code.code.lower()
    
    if any(x in name_lower for x in ['attention', 'mha', 'multihead']):
        return "attention"
    elif any(x in name_lower for x in ['transformer', 'encoder', 'decoder']):
        return "transformer"
    elif any(x in name_lower for x in ['conv', 'cnn', 'resnet', 'vgg']):
        return "cnn"
    elif any(x in name_lower for x in ['rnn', 'lstm', 'gru']):
        return "rnn"
    elif any(x in name_lower for x in ['mlp', 'linear', 'ffn', 'feedforward']):
        return "mlp"
    elif any(x in name_lower for x in ['norm', 'layer', 'batch']):
        return "normalization"
    elif any(x in name_lower for x in ['pool', 'downsample']):
        return "pooling"
    elif any(x in name_lower for x in ['embed', 'position']):
        return "embedding"
    elif 'conv2d' in code_lower or 'conv1d' in code_lower:
        return "cnn"
    elif 'attention' in code_lower:
        return "attention"
    else:
        return "other"


def create_output_folder(repo: str, source_file: str) -> Path:
    """
    Create organized output folder for a scraped source file.
    
    Structure:
        data/pycode/{repo_name}/{file_stem}/
            - source.py              (original file from GitHub)
            - extracted.py           (preview of extracted code)
            - scraped.jsonl          (training data)
            - synthetic.jsonl        (generated after running synthetic_challenges.py)
    """
    # Clean repo name for folder
    repo_name = repo.replace("/", "_").replace("-", "_").lower()
    
    # Get file stem (e.g., "x_transformers" from "x_transformers.py")
    file_stem = Path(source_file).stem.replace("-", "_").lower()
    
    # Create folder
    output_dir = Path("data/pycode") / repo_name / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_source_file(source_content: str, output_dir: Path) -> Path:
    """Copy original source file to output folder."""
    source_path = output_dir / "source.py"
    source_path.write_text(source_content)
    return source_path


def save_extracted_preview(extracted: list[ExtractedCode], output_dir: Path) -> Path:
    """
    Create a preview Python file showing all extracted code.
    
    This lets you manually verify extraction is correct before
    generating training data.
    """
    preview_path = output_dir / "extracted.py"
    
    lines = [
        '"""',
        'Extracted Code Preview',
        '=' * 60,
        'This file shows all code blocks extracted from source.py.',
        'Review this to verify extraction is correct before running',
        'the synthetic challenge generator.',
        '',
        f'Total extracted: {len(extracted)} blocks',
        '"""',
        '',
        'import torch',
        'import torch.nn as nn',
        'import torch.nn.functional as F',
        'from torch import Tensor',
        'from torch.nn import Module, ModuleList, Linear, LayerNorm',
        '',
        '',
        '# ' + '=' * 70,
        '# EXTRACTED CODE BLOCKS',
        '# ' + '=' * 70,
        '',
    ]
    
    for i, code in enumerate(extracted, 1):
        lines.append(f'# --- Block {i}: {code.name} ({code.code_type}) ---')
        lines.append(f'# From: {code.file_path}:{code.start_line}')
        lines.append(f'# Category: {categorize_code(code)}')
        lines.append('')
        lines.append(code.code)
        lines.append('')
        lines.append('')
    
    preview_path.write_text('\n'.join(lines))
    return preview_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract nn.Module code from GitHub repos"
    )
    parser.add_argument(
        "repo",
        help="GitHub repo (e.g., 'huggingface/pytorch-image-models' or full URL)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Specific paths within repo to search (files or directories)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: auto-generated from repo/file)",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines of code to include (default: 5)",
    )
    parser.add_argument(
        "--keep-repo",
        action="store_true",
        help="Keep cloned repo after extraction",
    )
    
    args = parser.parse_args()
    
    # Parse GitHub URL (handles full URLs with paths)
    repo, url_path = parse_github_url(args.repo)
    
    # Combine URL path with --paths argument
    paths = args.paths or []
    if url_path:
        paths = [url_path] + paths
    
    print("GitHub Code Scraper")
    print("=" * 60)
    print(f"Repo: {repo}")
    print(f"Paths: {paths or 'all'}")
    print()
    
    # Clone repo
    temp_dir = tempfile.mkdtemp()
    created_folders = []  # Track all folders created
    
    try:
        repo_path = clone_repo(repo, temp_dir)
        
        # Find Python files
        print("Finding Python files...")
        python_files = find_python_files(repo_path, paths if paths else None)
        print(f"Found {len(python_files)} Python files")
        
        # Process each file separately
        for file_path in python_files:
            rel_path = os.path.relpath(file_path, repo_path)
            print(f"\n{'─' * 60}")
            print(f"Processing: {rel_path}")
            
            # Read source content
            source_content = Path(file_path).read_text()
            
            # Extract code
            extracted = extract_from_file(file_path, repo_path)
            
            if not extracted:
                print(f"  No code blocks found, skipping...")
                continue
            
            print(f"  Extracted {len(extracted)} code blocks")
            
            # Create output folder
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = create_output_folder(repo, rel_path)
            
            created_folders.append(output_dir)
            
            # Save source file
            source_path = save_source_file(source_content, output_dir)
            print(f"  ✓ Saved source: {source_path}")
            
            # Save extracted preview
            preview_path = save_extracted_preview(extracted, output_dir)
            print(f"  ✓ Saved preview: {preview_path}")
            
            # Categorize
            categories = {}
            for code in extracted:
                cat = categorize_code(code)
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"  Categories: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")
            
            # Save JSONL
            jsonl_path = output_dir / "scraped.jsonl"
            saved, source_str = save_to_jsonl(
                extracted, 
                str(jsonl_path), 
                repo=repo, 
                min_lines=args.min_lines
            )
            print(f"  ✓ Saved scraped: {jsonl_path} ({saved} examples)")
            
            print(f"\n  📁 Output folder: {output_dir}")
            print(f"     - source.py      (original GitHub file)")
            print(f"     - extracted.py   (preview - review this!)")
            print(f"     - scraped.jsonl  (training data)")
            print(f"     - synthetic.jsonl (run synthetic_challenges.py to generate)")
        
    finally:
        if not args.keep_repo:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nRepo kept at: {temp_dir}")
    
    # Create combined scraped.jsonl at repo level
    if len(created_folders) > 1:
        repo_folder = created_folders[0].parent
        combined_path = repo_folder / "all_scraped.jsonl"
        
        total_examples = 0
        with open(combined_path, 'w') as combined:
            for folder in created_folders:
                scraped_file = folder / "scraped.jsonl"
                if scraped_file.exists():
                    with open(scraped_file) as f:
                        for line in f:
                            if line.strip():
                                combined.write(line)
                                total_examples += 1
        
        print(f"\n✓ Combined all scraped data: {combined_path} ({total_examples} examples)")
    
    print("\n" + "=" * 60)
    print(f"Done! Created {len(created_folders)} folder(s):")
    for folder in created_folders:
        print(f"  📁 {folder}")
    
    if len(created_folders) > 1:
        print(f"\n📦 Combined: {repo_folder}/all_scraped.jsonl")
    
    print("\nNext steps:")
    print("  1. Review extracted.py in each folder to verify extraction")
    print("  2. Generate synthetic prompts:")
    if len(created_folders) > 1:
        print(f"     # All at once:")
        print(f"     python scripts/synthetic_challenges.py --input {repo_folder}/all_scraped.jsonl --output {repo_folder}/all_synthetic.jsonl")
        print(f"     # Or individually:")
    for folder in created_folders[:3]:  # Show first 3
        print(f"     python scripts/synthetic_challenges.py --input {folder}/scraped.jsonl --output {folder}/synthetic.jsonl")
    if len(created_folders) > 3:
        print(f"     ... ({len(created_folders) - 3} more)")


if __name__ == "__main__":
    main()

