# Machine Learning & Deep Learning Course

Interactive lecture slides and laboratory notebooks for an AI, Machine Learning, and Deep Learning course.

- **Repository**: [https://git-ainf.aau.at/teaching/deeplearning.git](https://git-ainf.aau.at/teaching/deeplearning.git)

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
git clone https://git-ainf.aau.at/teaching/deeplearning.git
cd deeplearning

# Install dependencies and create .venv
uv sync
```

This single command (`uv sync`) will:
1. Automatically fetch the pinned Python runtime (Python 3.12).
2. Create the isolated `.venv` virtual environment.
3. Install all required packages (PyTorch, PyTorch Geometric, Scikit-Learn, JupyterLab, RISE slide extension, Matplotlib, Seaborn, Pandas, NLTK, etc.).

### 3. Launch the Course Environment

You can directly start JupyterLab or classic Jupyter Notebook with:

```bash
uv run jupyter lab
```
*or*
```bash
uv run jupyter notebook
```

### 4. Using with VS Code / Cursor / PyCharm

- Open the cloned `deeplearning` folder in VS Code, Cursor, or PyCharm.
- When opening any `.ipynb` notebook, click on the **Kernel / Python Environment** selector in the top right.
- Select the Python interpreter located inside `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows).

---

## 📽️ Presenting Slides (RISE / Reveal.js)

All lecture notebooks are equipped with Reveal.js / RISE slide metadata:
- In **JupyterLab**, click the **RISE Slide** button in the toolbar (or press `Alt + R`) to launch fullscreen interactive slides.
- Custom slide callouts (styled in `custom.html` and `rise.css`) are automatically loaded:
  - `<div class="myalert">...</div>` - Highlight key takeaway boxes.
  - `<div class="mydef">...</div>` - Formal definitions with blue sidebars.
  - `<div class="cite">...</div>` - Academic citations.
  - Fragments with `.step-fade-in-then-out` for animated slide reveals.

---

## ☁️ Google Colab Compatibility & Usage

The notebooks are fully compatible with Google Colab. Because Google Colab runs notebooks in an isolated `/content` directory without local repository files by default, use the following simple setup cell when opening notebooks in Colab:

### 1. Colab Bootstrap Cell

Add or run this snippet at the top of a notebook when running in Google Colab:

```python
# --- Google Colab Setup ---
import sys, os
if 'google.colab' in sys.modules:
    # 1. Clone the repository if not already present
    if not os.path.exists('deeplearning'):
        !git clone https://git-ainf.aau.at/teaching/deeplearning.git
    %cd deeplearning
    
    # 2. Install missing Colab dependencies
    !pip install -q torch-geometric wandb
```

### 2. Colab Suitability Summary

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
