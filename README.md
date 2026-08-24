# Machine Learning & Deep Learning Course

Interactive lecture slides and laboratory notebooks for an AI, Machine Learning, and Deep Learning course.

- **Repository**: [https://github.com/konstantin-schekotihin/dl_course.git](https://github.com/konstantin-schekotihin/dl_course.git)

---

## 📚 Course Structure

- **`01_Introduction/`**: Notation, Probability & Information Theory, Linear Algebra, Continuous Functions, Numerical Optimization.
- **`02_ML/`**: k-NN Classifiers, Naive Bayes, Support Vector Machines (SVM), Linear & Logistic Regression, Neural Networks, Unsupervised Learning.
- **`03_DL/`**: Artificial Neural Networks (ANN), Deep ANNs & TensorBoard, Convolutional Neural Networks (CNNs) & Architectures, Recurrent Neural Networks (RNN), Transformers & Attention, Graph Neural Networks (GNN via PyTorch Geometric).
- **`04_RL/`**: Reinforcement Learning (Multi-Armed Bandits).

---

## 🚀 Quick Start (Local Setup with `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager. `uv` automatically manages Python versions, virtual environments, and all dependencies without requiring manual Anaconda or Conda configurations.

### 1. Install `uv` (if not already installed)

- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  *(or via Homebrew: `brew install uv`)*

- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  *(or via winget: `winget install astral-sh.uv`)*

### 2. Clone the Repository & Set Up the Environment

Clone the repository and run `uv sync`:

```bash
# Clone the repository
git clone https://github.com/konstantin-schekotihin/dl_course.git
cd dl_course

# Install dependencies and create .venv
uv sync
```

This single command (`uv sync`) will:
1. Automatically fetch the pinned Python runtime (Python 3.12).
2. Create the isolated `.venv` virtual environment.
3. Install all required packages (PyTorch, PyTorch Geometric, Scikit-Learn, Jupyter Notebook, RISE slide extension, Matplotlib, Seaborn, Pandas, NLTK, etc.).

### 3. Launch the Course Environment

You can directly start Jupyter Notebook with:

```bash
source .venv/bin/activate
jupyter notebook
```
*or directly with `uv`:*
```bash
uv run jupyter notebook
```

### 4. Using with VS Code / Cursor / PyCharm

- Open the cloned `dl_course` folder in VS Code, Cursor, or PyCharm.
- When opening any `.ipynb` notebook, click on the **Kernel / Python Environment** selector in the top right.
- Select the Python interpreter located inside `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows).

> [!NOTE]
> **Formula Rendering in VS Code (KaTeX) & Colab:**
> VS Code (via KaTeX) and Google Colab (before running code cells) render Markdown cells statically. If you need formula macros registered immediately in Markdown without executing Python cells, add this standard HTML-wrapped Markdown cell at the top of the notebook:
> ```html
> <div style="display:none">
> $$
> \def\rvar#1{\mathrm{#1}}
> \def\rvec#1{\mathbf{#1}}
> \def\vec#1{\boldsymbol{#1}}
> \def\tens#1{\boldsymbol{\mathsf{#1}}}
> \def\tensel#1{\mathsf{#1}}
> \def\st#1{\mathcal{#1}}
> \def\diag#1{\mathrm{diag}(\vec{#1})}
> $$
> </div>
> ```

---

## 📽️ Presenting Slides (RISE / Reveal.js)

All lecture notebooks are equipped with Reveal.js / RISE slide metadata:
- In **Jupyter Notebook**, click the **RISE Slide** button in the toolbar (or press `Alt + R`) to launch fullscreen interactive slides.
- Custom slide callouts (styled in `custom.html` and `rise.css`) are automatically loaded:
  - `<div class="myalert">...</div>` - Highlight key takeaway boxes.
  - `<div class="mydef">...</div>` - Formal definitions with blue sidebars.
  - `<div class="cite">...</div>` - Academic citations.
  - Fragments with `.step-fade-in-then-out` for animated slide reveals.

---

## ☁️ Google Colab Compatibility & Usage

Because the repository is hosted on GitHub, you can open any notebook directly in Google Colab with native 1-click integration:

### 1. Open Directly via Colab's GitHub Tab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Open notebook** and choose the **GitHub** tab.
3. Enter `konstantin-schekotihin/dl_course` (or paste `https://github.com/konstantin-schekotihin/dl_course`).
4. Select any `.ipynb` notebook from the course to open it immediately.

*(Alternatively, construct a direct link: `https://colab.research.google.com/github/konstantin-schekotihin/dl_course/blob/master/<module>/<notebook>.ipynb`)*

### 2. Automatic Colab Setup (Pre-configured)

All course notebooks already include automatic Colab bootstrap logic in their very first cell:
- **On Google Colab:** It automatically clones the course repository in the background, sets the module working directory, and installs any missing packages (`torch-geometric`, `wandb`).
- **On Local Jupyter / VS Code:** The Colab check is automatically skipped and executes standard `%run ../init.py` without network calls.

```python
# Setup for Google Colab (pre-configured in cell 1)
import sys, os
if 'google.colab' in sys.modules:
    if not os.path.exists('dl_course'):
        !git clone -q --depth 1 https://github.com/konstantin-schekotihin/dl_course.git
    %cd -q /content/dl_course/01_Introduction
    !pip install -q torch-geometric wandb

%run ../init.py
%matplotlib inline
```

---

### Colab Suitability Summary

| Feature / Requirement | Local (`uv`) | Google Colab | Colab Notes |
|---|---|---|---|
| **PyTorch & Torchvision** | Pre-configured in `.venv` | Pre-installed | Native GPU acceleration available (T4 / A100) |
| **PyTorch Geometric (PyG)** | Installed via `uv sync` | `!pip install torch-geometric` | Needed for `03_DL/06-Graphs.ipynb` |
| **Scikit-Learn, SciPy, Pandas** | Installed via `uv sync` | Pre-installed | Fully compatible |
| **`init.py` & Custom CSS** | Loaded via `%run ../init.py` | Auto-fallback | `init.py` handles missing paths & injects CSS inline |
| **Datasets & Local Images** | Available locally | Available after clone | Clone repo to access `data/` and `images/` |
| **Slideshow Presentation** | Full interactive RISE | Focus / Presentation View | Colab offers section folding and presentation view |

---

## 🛠️ Maintenance & Dependency Updates

To add new dependencies or update existing ones:
```bash
# Add a new package
uv add <package-name>

# Update locked dependencies
uv lock --upgrade
uv sync
```
