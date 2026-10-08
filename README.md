# Machine Learning & Deep Learning Course

Interactive lecture slides and laboratory notebooks for an AI, Machine Learning, and Deep Learning curriculum.

- **Repository**: [https://github.com/konstantin-schekotihin/dl_course.git](https://github.com/konstantin-schekotihin/dl_course.git)

---

## 📚 Courses & Repository Structure

This repository hosts interactive lecture slides, laboratory notebooks, and course materials organized into two distinct academic tracks and specialized electives:

- 1. [ML-DL/] — Machine Learning & Deep Learning
- 2. [AI-ML/] — AI and Machine Learning
- 3. [Extended/] — Specialized & Advanced Electives

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
uv run jupyter notebook
```
*or activate the virtual environment directly:*
```bash
source .venv/bin/activate
jupyter notebook
```

### 4. Using with VS Code / Cursor / PyCharm

- Open the cloned `dl_course` folder in VS Code, Cursor, or PyCharm.
- When opening any `.ipynb` notebook, click on the **Kernel / Python Environment** selector in the top right.
- Select the Python interpreter located inside `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows).

---

## ☁️ Google Colab Usage & Drive Setup

Students can run and edit all notebooks directly in Google Colab with free GPU acceleration (NVIDIA T4 / A100).

### 🚀 1-Click Setup Utility (`colab_setup.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantin-schekotihin/dl_course/blob/master/colab_setup.ipynb)

We provide an interactive **`colab_setup.ipynb`** notebook to automate the entire Google Drive workflow:
1. **One-Time Clone to Google Drive:** Mounts your Google Drive and clones `dl_course` so your work, exercise solutions, and model weights (`.pth`) are saved permanently.
2. **Automatic Syncing:** Re-running `colab_setup.ipynb` pulls the latest lecture slides, datasets, and updates from GitHub (`git pull --autostash`).
3. **Hardware & GPU Verification:** Checks PyTorch version, CUDA availability, and installed packages.

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
