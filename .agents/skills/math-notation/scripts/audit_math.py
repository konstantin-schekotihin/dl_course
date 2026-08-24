#!/usr/bin/env python3
"""
Audits and maintains mathematical notation and Cell 0 macro definitions
across all Jupyter notebooks in the course repository.
"""

import os
import sys
import json
import argparse

REQUIRED_MACROS = [
    r"\newcommand{\rvar}[1]{\mathrm{#1}}",
    r"\newcommand{\rvec}[1]{\mathbf{#1}}",
    r"\newcommand{\vec}[1]{\boldsymbol{#1}}",
    r"\newcommand{\tens}[1]{\boldsymbol{\mathsf{#1}}}",
    r"\newcommand{\tensel}[1]{\mathsf{#1}}",
    r"\newcommand{\st}[1]{\mathcal{#1}}",
    r"\newcommand{\diag}[1]{\mathrm{diag}(\vec{#1})}",
]

HIDDEN_MATH_CELL = {
    "cell_type": "markdown",
    "metadata": {
        "slideshow": {
            "slide_type": "skip"
        }
    },
    "source": [
        "<div style=\"display:none\">\n",
        "$$\n",
        "\\newcommand{\\rvar}[1]{\\mathrm{#1}}\n",
        "\\newcommand{\\rvec}[1]{\\mathbf{#1}}\n",
        "\\newcommand{\\vec}[1]{\\boldsymbol{#1}}\n",
        "\\newcommand{\\tens}[1]{\\boldsymbol{\\mathsf{#1}}}\n",
        "\\newcommand{\\tensel}[1]{\\mathsf{#1}}\n",
        "\\newcommand{\\st}[1]{\\mathcal{#1}}\n",
        "\\newcommand{\\diag}[1]{\\mathrm{diag}(\\vec{#1})}\n",
        "$$\n",
        "</div>"
    ]
}

def find_notebooks(repo_root):
    notebooks = []
    for root, dirs, files in os.walk(repo_root):
        if any(ignored in root for ignored in ['.venv', '.git', '.ipynb_checkpoints', 'node_modules']):
            continue
        for file in sorted(files):
            if file.endswith('.ipynb'):
                notebooks.append(os.path.join(root, file))
    return sorted(notebooks)

def audit_notebook(nb_path, repo_root, fix=False):
    rel_path = os.path.relpath(nb_path, repo_root)
    with open(nb_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            return False, [f"JSON Decode Error: {e}"]
            
    cells = nb.get('cells', [])
    if not cells:
        return False, ["Notebook has no cells"]
        
    issues = []
    first_cell = cells[0]
    first_source = "".join(first_cell.get('source', []))
    
    # 1. Check if first cell is a markdown hidden math cell
    is_markdown = first_cell.get('cell_type') == 'markdown'
    has_hidden_div = '<div style="display:none">' in first_source
    has_skip_metadata = first_cell.get('metadata', {}).get('slideshow', {}).get('slide_type') == 'skip'
    
    if not is_markdown or not has_hidden_div:
        issues.append("Cell 0 is missing the hidden MathJax <div style=\"display:none\"> definition block.")
    elif not has_skip_metadata:
        issues.append("Cell 0 is missing 'slideshow: {slide_type: skip}' metadata.")
        
    # Check for all required macros
    missing_macros = []
    for macro in [r"\rvar", r"\rvec", r"\vec", r"\tens", r"\tensel", r"\st", r"\diag"]:
        if macro not in first_source:
            missing_macros.append(macro)
            
    if missing_macros:
        issues.append(f"Cell 0 is missing required macros: {', '.join(missing_macros)}")
        
    # Check for forbidden Pandoc ::: {.hidden} syntax
    if "::: {.hidden}" in first_source:
        issues.append("Cell 0 uses invalid Pandoc '::: {.hidden}' syntax instead of HTML '<div style=\"display:none\">'.")
        
    if fix and issues:
        # Fix cell 0
        if is_markdown and (has_hidden_div or "::: {.hidden}" in first_source):
            cells[0] = dict(HIDDEN_MATH_CELL)
        else:
            cells = [dict(HIDDEN_MATH_CELL)] + cells
            
        nb['cells'] = cells
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        return True, ["FIXED: Cell 0 updated with standard hidden MathJax definitions."]
        
    return len(issues) == 0, issues

def main():
    parser = argparse.ArgumentParser(description="Audit and maintain course math notation.")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--fix", action="store_true", help="Automatically fix missing or malformed Cell 0 math blocks")
    args = parser.parse_args()
    
    repo_root = os.path.abspath(args.repo_root)
    notebooks = find_notebooks(repo_root)
    
    print(f"Auditing {len(notebooks)} course notebooks for math notation compliance...\n")
    
    all_ok = True
    for nb in notebooks:
        rel = os.path.relpath(nb, repo_root)
        ok, issues = audit_notebook(nb, repo_root, fix=args.fix)
        if not ok:
            all_ok = False
            print(f"❌ {rel}:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            status = "🔧 FIXED" if args.fix and issues else "✅ OK"
            print(f"{status}: {rel}")
            
    print("\n" + ("=" * 50))
    if all_ok:
        print("🎉 All notebooks comply with course math notation standards!")
        sys.exit(0)
    else:
        print("⚠️ Issues found. Run with --fix to automatically resolve them.")
        sys.exit(1)

if __name__ == "__main__":
    main()
