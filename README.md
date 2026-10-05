# Machine Learning & Deep Learning Course

Interactive lecture slides and laboratory notebooks for an AI, Machine Learning, and Deep Learning course.

- **Repository**: [https://github.com/konstantin-schekotihin/dl_course.git](https://github.com/konstantin-schekotihin/dl_course.git)

---

## 📚 Courses & Repository Structure

This repository hosts interactive lecture slides, laboratory notebooks, and course materials organized into two distinct academic tracks and specialized electives:

### 🎓 1. [ML-DL/](file:///Users/kostya/Documents/Teaching/ML-DL/ML-DL/README.md) — Machine Learning & Deep Learning
*Target Audience: Master's in Computer Science & Artificial Intelligence.*
- **`01_foundations/`**: Mathematical Foundations (Bishop & Bishop 2024, Ch 1–3) — Notation, Probability Theory, Information Theory & Entropy, Linear Algebra, Continuous Functions, and MLOps.
- **`02_deep_learning/`**: Deep Neural Architectures (Bishop & Bishop 2024, Ch 4–13) — Single-layer Networks (ANN), Deep MLPs & TensorBoard, CNNs & Vision Backbones, Recurrent Networks (RNN/LSTM), Transformers & Self-Attention, Graph Neural Networks (GNN via PyG), and Stochastic Numerical Optimization.

### 💼 2. [AI-ML/](file:///Users/kostya/Documents/Teaching/ML-DL/AI-ML/README.md) — AI and Machine Learning
*Target Audience: Applied AI, Data Science & Management (Non-CS Students).*
- Complete 10-module applied curriculum structured around the CRISP-DM lifecycle:
  1. AI Landscape, Paradigms & MLOps
  2. k-NN & Complete ML Project Lifecycle with Weights & Biases (W&B)
  3. Simple & Multiple Linear Regression
  4. Logistic Regression & Classification Diagnostics
  5. Neural Networks Intuition & Backpropagation
  6. Decision Tree Ensembles (Bagging, Random Forests, Boosting)
  7. Support Vector Machines & Kernels
  8. Unsupervised Learning (Clustering & Dimensionality Reduction)
  9. Naive Bayes & Text Classification
  10. AI Coding Agents in Data Science & Machine Learning
- Includes [AI-ML/modalities.md](file:///Users/kostya/Documents/Teaching/ML-DL/AI-ML/modalities.md) (grading scale, group project, deadlines, and GenAI policy).

### 🔬 3. [Extended/](file:///Users/kostya/Documents/Teaching/ML-DL/Extended/README.md) — Specialized & Advanced Electives
- **`01_reinforcement_learning/`**: Reinforcement Learning (Multi-Armed Bandits, Exploration vs. Exploitation, $\epsilon$-greedy, UCB, Thompson Sampling).

### 📦 4. `shared/` — Canonical Asset Repository
- **`shared/data/`**: Centralized datasets (`Advertising.csv`, `Default.csv`, `USArrests.csv`, `breast-cancer.csv`, `MNIST`).
- **`shared/images/`**: Deduplicated canonical figures and illustrations.
- **`shared/styles/`**: RISE presentation styles (`rise.css` and `custom.html`).

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
> \newcommand{\rvar}[1]{\mathrm{#1}}
> \newcommand{\rvec}[1]{\mathbf{#1}}
> \newcommand{\vec}[1]{\pmb{#1}}
> \newcommand{\tens}[1]{\pmb{\mathsf{#1}}}
> \newcommand{\tensel}[1]{\mathsf{#1}}
> \newcommand{\st}[1]{\mathcal{#1}}
> \newcommand{\diag}[1]{\mathrm{diag}(\vec{#1})}
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

## ☁️ Google Colab Usage & Drive Setup

Students can run and edit all notebooks directly in Google Colab with free GPU acceleration (NVIDIA T4 / A100).

### 🚀 1-Click Setup Utility (`colab_setup.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantin-schekotihin/dl_course/blob/master/colab_setup.ipynb)

We provide an interactive **`colab_setup.ipynb`** notebook to automate the entire Google Drive workflow:
1. **One-Time Clone to Google Drive:** Mounts your Google Drive and clones `dl_course` so your work, exercise solutions, and model weights (`.pth`) are saved permanently.
2. **Automatic Syncing:** Re-running `colab_setup.ipynb` pulls the latest lecture slides, datasets, and updates from GitHub (`git pull --autostash`).
3. **Hardware & GPU Verification:** Checks PyTorch version, CUDA availability, and installed packages.

---

### Opening Lecture Notebooks in Colab

- **From Google Drive (Recommended):** After running `colab_setup.ipynb`, open [Google Drive](https://drive.google.com/), navigate to `MyDrive/dl_course/`, right-click any `.ipynb` notebook $\rightarrow$ **Open with $\rightarrow$ Google Colaboratory**.
- **Directly from GitHub:** In [Google Colab](https://colab.research.google.com/), go to **File $\rightarrow$ Open notebook $\rightarrow$ GitHub** tab, search `konstantin-schekotihin/dl_course`, and select any notebook.

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
