---
name: math-notation
description: >-
  Maintains, audits, and enforces consistent mathematical notation and LaTeX macro
  definitions across course Jupyter notebooks for Jupyter Notebook 6, RISE slides,
  and Google Colab. Use when adding new notebooks, editing mathematical formulas,
  auditing LaTeX rendering, or troubleshooting MathJax/LaTeX errors.
---

# Math Notation Maintenance Skill

This skill defines the mathematical notation conventions, LaTeX macro standards, and cross-platform compatibility rules for the Machine Learning & Deep Learning course repository.

---

## 1. Course Mathematical Notation Standards

All notebooks in `01_Introduction/`, `02_ML/`, `03_DL/`, and `04_RL/` adhere to the standard mathematical notation summarized below:

| Concept | Shorthand Macro | LaTeX Expansion | Rendered Result | Example Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Scalar variable** | `$x$` | `$x$` | $x$ | `$x \in \mathbb{R}$` |
| **Scalar random variable** | `\rvar{x}` | `\mathrm{x}` | $\mathrm{x}$ | `$\rvar{x} \sim P$` |
| **Vector** | `\vec{x}` | `\boldsymbol{x}` | $\boldsymbol{x}$ | `$\vec{x} \in \mathbb{R}^n$` |
| **Vector random variable** | `\rvec{x}` | `\mathbf{x}` | $\mathbf{x}$ | `$\rvec{x} \sim \mathcal{N}(\vec{\mu}, \vec{\Sigma})$` |
| **Matrix** | `\vec{X}` or `\mathbf{X}` | `\boldsymbol{X}` / `\mathbf{X}` | $\boldsymbol{X}$ | `$\vec{X} \in \mathbb{R}^{m \times n}$` |
| **Tensor (3D+)** | `\tens{X}` | `\boldsymbol{\mathsf{X}}` | $\boldsymbol{\mathsf{X}}$ | `$\tens{X} \in \mathbb{R}^{h \times w \times c}$` |
| **Tensor element** | `\tensel{A}_{i,j,k}` | `\mathsf{A}_{i,j,k}` | $\mathsf{A}_{i,j,k}$ | `$\tensel{A}_{i,j,k}$` |
| **Set** | `\st{X}` | `\mathcal{X}` | $\mathcal{X}$ | `$\st{X} = \{x_1, \dots, x_n\}$` |
| **Diagonal matrix** | `\diag{a}` | `\mathrm{diag}(\boldsymbol{a})` | $\mathrm{diag}(\boldsymbol{a})$ | `$\diag{a}$` |

---

## 2. Universal Macro Definition Standard

To ensure mathematical formulas render seamlessly across **Google Colab** and **classic Jupyter Notebook 6 + RISE slides** without yellow warnings or visual artifacts:

### Rule A: Cell 0 in Every Notebook
Every `.ipynb` file in the repository must have a hidden Markdown cell as **Cell 0** containing:

```html
<div style="display:none">
$$
\newcommand{\rvar}[1]{\mathrm{#1}}
\newcommand{\rvec}[1]{\mathbf{#1}}
\newcommand{\vec}[1]{\boldsymbol{#1}}
\newcommand{\tens}[1]{\boldsymbol{\mathsf{#1}}}
\newcommand{\tensel}[1]{\mathsf{#1}}
\newcommand{\st}[1]{\mathcal{#1}}
\newcommand{\diag}[1]{\mathrm{diag}(\vec{#1})}
$$
</div>
```

With slide metadata:
```json
"metadata": {
  "slideshow": {
    "slide_type": "skip"
  }
}
```

### Rule B: Global Python Runtime Initialization ([`init.py`](file:///Users/kostya/Documents/Teaching/ML-DL/init.py))
[`init.py`](file:///Users/kostya/Documents/Teaching/ML-DL/init.py) must register the exact matching definitions during Python execution:

```python
# Course LaTeX macros
display(Latex("""$$
\\newcommand{\\rvar}[1]{\\mathrm{#1}}
\\newcommand{\\rvec}[1]{\\mathbf{#1}}
\\newcommand{\\vec}[1]{\\boldsymbol{#1}}
\\newcommand{\\tens}[1]{\\boldsymbol{\\mathsf{#1}}}
\\newcommand{\\tensel}[1]{\\mathsf{#1}}
\\newcommand{\\st}[1]{\\mathcal{#1}}
\\newcommand{\\diag}[1]{\\mathrm{diag}(\\vec{#1})}
$$"""))
```

---

## 3. Platform Compatibility Mechanics & Critical Rules

### 1. Google Colab (MathJax 2.7)
* **Parent DOM Rendering:** Google Colab renders Markdown cells in the parent browser DOM while sandboxing Python outputs inside isolated `<iframe>` elements.
* **Cell 0 Necessity:** Because iframe outputs cannot inject MathJax definitions into parent Markdown cells, Cell 0's Markdown block is mandatory for Colab to render equations on page load.
* **`\newcommand` Requirement:** Colab's MathJax 2.7 build does not load the optional `begingroup.js` extension by default; therefore, always use `\newcommand` or `\def`.

### 2. Jupyter Notebook 6 & RISE Slides (MathJax 2/3)
* **Never use Pandoc `::: {.hidden}` syntax:** Jupyter 6 does not parse Pandoc container divs in Markdown and displays the literal `::: {.hidden}` string along with yellow MathJax warning boxes. Always use `<div style="display:none">`.
* **Always set `"slideshow": {"slide_type": "skip"}` on Cell 0:** This instructs the RISE slide builder to completely omit Cell 0, preventing empty blank slides during live lectures.

### 3. Typography Rules
* **Use `\mathrm` instead of `\textrm`:** `\mathrm` is math-mode roman and renders consistently across all mathematical rendering engines.
* **Subscript Braces:** Always enclose multi-character or macro subscripts in curly braces (e.g. `$\sigma_{\rvar{x}}$`, never `$\sigma_\rvar{x}$`).
* **Inline Math Delimiters:** Avoid placing double display delimiters `$$...$$` inside inline markdown bullet lists; use single dollar signs `$...$`.

---

## 4. Maintenance & Audit Workflow

When adding a new notebook or modifying equations, run the automated compliance auditor:

```bash
# 1. Audit all notebooks for math compliance
uv run python .agents/skills/math-notation/scripts/audit_math.py

# 2. Automatically fix missing or malformed Cell 0 blocks
uv run python .agents/skills/math-notation/scripts/audit_math.py --fix
```
