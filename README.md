# Machine Learning & Deep Learning Course

Interactive lecture slides and laboratory notebooks for an AI, Machine Learning, and Deep Learning curriculum.

- **Repository**: [https://github.com/konstantin-schekotihin/dl_course.git](https://github.com/konstantin-schekotihin/dl_course.git)

---

## 📚 Courses & Repository Structure

This repository hosts interactive lecture slides, laboratory notebooks, and course materials organized into two distinct academic tracks and specialized electives:

### 🎓 1. [ML-DL/](file:///Users/kostya/Documents/Teaching/ML-DL/ML-DL/README.md) — Machine Learning & Deep Learning
*Target Audience: Master's in Computer Science & Artificial Intelligence.*  
*Primary Reference: Christopher M. Bishop & Hugh Bishop (2024), **Deep Learning: Foundations and Concepts**, Springer.*

- **`01_foundations/`**: Mathematical Foundations (Bishop & Bishop 2024, Ch. 1–3)
  1. [`00_notation.ipynb`](ML-DL/01_foundations/00_notation.ipynb): Pure native LaTeX mathematical typography, scalar/vector/matrix conventions, and Bishop probability notation.
  2. [`01_Introduction.ipynb`](ML-DL/01_foundations/01_Introduction.ipynb): Deep learning revolution, symbolic vs. connectionist paradigms, learning tasks, and inductive biases.
  3. [`02_probability.ipynb`](ML-DL/01_foundations/02_probability.ipynb): Probability theory, sum and product rules, Bayes' theorem, densities, and bias-variance decomposition.
  4. [`03_information.ipynb`](ML-DL/01_foundations/03_information.ipynb): Information theory, Shannon entropy, differential entropy, cross-entropy, and Kullback-Leibler (KL) divergence.
  5. [`04_algebra.ipynb`](ML-DL/01_foundations/04_algebra.ipynb): Linear algebra, vector geometry, orthogonal projections, pseudo-inverse, eigendecomposition, SVD, and Einstein summation (`torch.einsum`).
  6. [`05_functions.ipynb`](ML-DL/01_foundations/05_functions.ipynb): Multivariable calculus, Jacobians, Hessians, activation functions, softplus identities, and canonical link theorem.
  7. [`06_mlops.ipynb`](ML-DL/01_foundations/06_mlops.ipynb): Machine Learning Operations (MLOps), data provenance, experiment reproducibility, and pipeline hygiene.

- **`02_deep_learning/`**: Deep Neural Architectures (Bishop & Bishop 2024, Ch. 4–13)
  1. [`04_regression.ipynb`](ML-DL/02_deep_learning/04_regression.ipynb): Single-layer Networks: Regression (Linear basis functions, OLS normal equations, regularized least squares, sequential Bayesian updates).
  2. [`05_classification.ipynb`](ML-DL/02_deep_learning/05_classification.ipynb): Single-layer Networks: Classification (Discriminant functions, Fisher's linear discriminant, Perceptron, logistic sigmoid, multiclass softmax).
  3. [`06_deep_networks.ipynb`](ML-DL/02_deep_learning/06_deep_networks.ipynb): Deep Neural Networks (Multilayer perceptrons, universal approximation, representations, FashionMNIST benchmarking).
  4. [`07_gradient_descent.ipynb`](ML-DL/02_deep_learning/07_gradient_descent.ipynb): Gradient Descent (Optimization landscapes, batch vs. mini-batch SGD, Polyak/Nesterov momentum, RMSprop, Adam).
  5. [`08_backpropagation.ipynb`](ML-DL/02_deep_learning/08_backpropagation.ipynb): Backpropagation (Computational graphs, reverse-mode automatic differentiation, vector chain rule, pure NumPy backprop verified against PyTorch Autograd).
  6. [`09_regularization.ipynb`](ML-DL/02_deep_learning/09_regularization.ipynb): Regularization ($L_2$ weight decay, early stopping, data augmentation, dropout, batch normalization).
  7. [`10_convolutional_networks.ipynb`](ML-DL/02_deep_learning/10_convolutional_networks.ipynb): Convolutional Networks (Spatial filtering, weight sharing, translational equivariance, receptive fields, modern CNN architectures).
  8. [`11_sequence_models.ipynb`](ML-DL/02_deep_learning/11_sequence_models.ipynb): Sequence Models (Recurrent neural networks, vanishing/exploding gradients, LSTM, GRU, sequence forecasting).
  9. [`12_transformers.ipynb`](ML-DL/02_deep_learning/12_transformers.ipynb): Transformers (Scaled dot-product attention, multi-head self-attention, positional encodings, encoder-decoder architectures).
  10. [`13_graph_neural_networks.ipynb`](ML-DL/02_deep_learning/13_graph_neural_networks.ipynb): Graph Neural Networks (Permutation equivariance, message passing neural networks, GCN via PyTorch Geometric).

---

### 💼 2. [AI-ML/](file:///Users/kostya/Documents/Teaching/ML-DL/AI-ML/README.md) — AI and Machine Learning
*Target Audience: Applied AI, Data Science & Management (Non-CS Students).*  
*Curriculum structure aligned with the CRISP-DM lifecycle and practical experiment tracking.*

- **Applied Curriculum**:
  1. [`00_introduction.ipynb`](AI-ML/00_introduction.ipynb): AI Landscape, Paradigms, Learning Tasks $(T, P, E)$, and MLOps.
  2. [`01_classifiers-kNN.ipynb`](AI-ML/01_classifiers-kNN.ipynb): $k$-NN & Complete ML Project Lifecycle with Weights & Biases (W&B) *(Project Template)*.
  3. [`02_regression.ipynb`](AI-ML/02_regression.ipynb): Simple & Multiple Linear Regression, Residuals, $R^2$, MSE, and Ridge/Lasso Regularization.
  4. [`03_logistic_regression.ipynb`](AI-ML/03_logistic_regression.ipynb): Logistic Regression, Sigmoid Activation, Decision Boundaries, and ROC-AUC.
  5. [`04_NN.ipynb`](AI-ML/04_NN.ipynb): Neural Network Intuition, Multilayer Perceptrons, and Loss Surface Optimization.
  6. [`05_ensembles.ipynb`](AI-ML/05_ensembles.ipynb): Decision Tree Ensembles (Bagging, Random Forests, AdaBoost, Gradient Boosting, W&B Logging).
  7. [`06_SVM.ipynb`](AI-ML/06_SVM.ipynb): Support Vector Machines, Maximum Margin Hyperplanes, Slack Variables, and RBF Kernels.
  8. [`07_unsupervised.ipynb`](AI-ML/07_unsupervised.ipynb): Unsupervised Structure Discovery, $k$-Means, Hierarchical Clustering, and PCA.
  9. [`08_ml_agents.ipynb`](AI-ML/08_ml_agents.ipynb): AI Coding Agents in Machine Learning: Specification-Driven Engineering, TDD Invariants, and Defensio Readiness.
  10. [`aux1_BN.ipynb`](AI-ML/aux1_BN.ipynb) *(Auxiliary)*: Bayesian Classifiers, Conditional Independence, Laplace Smoothing, and Text Classification.
- **Course Policies & Modalities**: [AI-ML/modalities.md](file:///Users/kostya/Documents/Teaching/ML-DL/AI-ML/modalities.md).

---

### 🔬 3. [Extended/](file:///Users/kostya/Documents/Teaching/ML-DL/Extended/README.md) — Specialized & Advanced Electives
- **`01_reinforcement_learning/`**: Reinforcement Learning ([`07_RL.ipynb`](Extended/01_reinforcement_learning/07_RL.ipynb))
  - Multi-Armed Bandits, Exploration vs. Exploitation, $\epsilon$-greedy, Upper-Confidence-Bound (UCB), Thompson Sampling, and Introduction to Deep Q-Networks (DDQN).

---

### 📦 4. `shared/` — Canonical Assets & Shared Utilities
- **`shared/helpers.py`**: Shared helper core powering course-specific helper modules. Provides unified theme configuration (`setup_theme()`), modular visualization routines, and asset resolution functions:
  - `hp.get_file("dataset.csv")` — resolves local assets with self-healing Colab download fallback.
  - `hp.get_data_path("")` — resolves dataset directory roots for torchvision loaders (`FashionMNIST`, `CIFAR10`).
  - `hp.get_url("...")` — provides remote asset URLs.
- **`shared/data/`**: Centralized datasets (`Advertising.csv`, `Default.csv`, `USArrests.csv`, `breast-cancer.csv`, `model_example.csv`).
- **`shared/images/`**: Deduplicated canonical figures and illustrations.
- **`shared/styles/`**: RISE presentation styles (`rise.css`).

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

## 📐 Mathematical Notation Standards (Bishop Doctrine)

All mathematics across course notebooks strictly adheres to Christopher M. Bishop & Hugh Bishop (2024), *Deep Learning: Foundations and Concepts*:
- **Pure Native LaTeX**: Standard primitives only (`\mathbf`, `\boldsymbol`, `\mathbb`, `\mathcal`, `\mathrm`). No custom macros or hidden Cell 0 definitions are used; all formulas render natively across Jupyter, VS Code (KaTeX), and Google Colab.
- **Variables & Distributions**: Random variables share standard lowercase typography ($x \sim p(x)$, $\mathbf{x} \sim p(\mathbf{x})$).
- **Vectors & Matrices**: Column vectors $\mathbf{x}, \boldsymbol{\theta} \in \mathbb{R}^D$; design matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$; target vectors $\mathbf{t} \in \mathbb{R}^N$.
- **Expectation & Variance**: $\mathbb{E}[x]$, $\mathrm{var}[x]$, $\mathrm{cov}[\mathbf{x}, \mathbf{y}]$.
- **Rich Mathematical Output**: All code output displays math using `display(Markdown(...))` with LaTeX formatting (e.g. `$\mathbf{w}^\star$`, $\mathrm{MSE}$) rather than ASCII approximations.

---

## 📽️ Presenting Slides (RISE / Reveal.js)

All lecture notebooks are equipped with Reveal.js / RISE slide metadata:
- In **Jupyter Notebook**, click the **RISE Slide** button in the toolbar (or press `Alt + R`) to launch fullscreen interactive slides.
- Target viewport heights are strictly calibrated ($\le 820$px) for Full HD 1080p projectors.
- Slide callouts use native Markdown syntax styled with clean visual alerts:
  - `> ⚠️ **Important:** ...` — Critical warnings and common pitfalls.
  - `> 💡 **Key Takeaway:** ...` — Core architectural and conceptual summaries.
  - `> 📘 **Definition: ...**` — Formal mathematical and theoretical definitions.
  - `<small>📖 *Source: [Authors (Year). Title. Venue](https://doi.org/...)*</small>` — Academic literature citations.

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
