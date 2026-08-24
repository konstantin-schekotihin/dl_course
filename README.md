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
> **Formula Rendering in Colab / VS Code:**
> If you open notebooks directly in Google Colab or VS Code before running the initialization code, LaTeX shorthand macros (`\vec`, `\tens`, `\diag`, `\st`, `\rvar`, `\rvec`, `\tensel`) can be registered by placing this standard Markdown math cell at the top of the notebook:
> ```markdown
> $$
> \newcommand{\vec}[1]{\mathbf{#1}}
> \newcommand{\tens}[1]{\mathsf{#1}}
> \newcommand{\diag}[1]{\mathrm{diag}(#1)}
> \newcommand{\st}[1]{\mathcal{#1}}
> \newcommand{\rvar}[1]{\mathrm{#1}}
> \newcommand{\rvec}[1]{\mathbf{#1}}
> \newcommand{\tensel}[1]{\mathcal{#1}}
> $$
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

### 2. Colab Setup Cells (In-Notebook)

When opening notebooks in Google Colab:

**A. Markdown Math Cell (for immediate formula rendering):**
```markdown
$$
\newcommand{\vec}[1]{\mathbf{#1}}
\newcommand{\tens}[1]{\mathsf{#1}}
\newcommand{\diag}[1]{\mathrm{diag}(#1)}
\newcommand{\st}[1]{\mathcal{#1}}
\newcommand{\rvar}[1]{\mathrm{#1}}
\newcommand{\rvec}[1]{\mathbf{#1}}
\newcommand{\tensel}[1]{\mathcal{#1}}
$$
```

**B. Code Setup Cell (to access repository assets and dependencies):**
```python
# --- Google Colab Setup ---
import sys, os
if 'google.colab' in sys.modules:
    # 1. Clone repository to access init.py, data, and images
    if not os.path.exists('dl_course'):
        !git clone https://github.com/konstantin-schekotihin/dl_course.git
    
    # 2. Change working directory into the module folder so %run ../init.py works
    # Example for 01_Introduction notebooks (adjust folder name accordingly):
    %cd /content/dl_course/01_Introduction
    
    # 3. Install packages not pre-installed on Colab
    !pip install -q torch-geometric wandb
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
