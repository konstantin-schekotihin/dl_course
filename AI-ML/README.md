# 🤖 AI and Machine Learning Course

Welcome to the **AI and Machine Learning** course. This curriculum is designed to develop practical analytical and engineering skills:
- Formulating real-world learning tasks $(T, P, E)$.
- Selecting appropriate machine learning algorithms based on problem characteristics.
- Rigorously training, tuning, and evaluating models with leak-free pipelines.
- Tracking experiments with Weights & Biases (W&B).
- Interpreting results to make informed domain decisions.

---

## 🚀 Running the Notebooks

You can run these notebooks in two ways:
1. **Google Colab (Recommended for zero-install quickstart)**
2. **Local Environment via `uv` (Recommended for project work)**

---

### Option 1: 1-Click Google Colab
Every notebook includes an **Open in Colab** badge at the very top.
- Simply click the badge to open the notebook directly in your web browser.
- Datasets are automatically loaded via direct streaming—no manual file uploads or Drive mounting required!

---

### Option 2: Local Setup via `uv` (Fast & Reproducible)
To run the notebooks locally on Windows, macOS, or Linux:

#### 1. Install `uv`
If you haven't installed `uv` yet:
- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

#### 2. Synchronize Environment
From the repository root directory, run:
```bash
uv sync
```

#### 3. Launch Jupyter
Start Jupyter inside the synchronized environment:
```bash
uv run jupyter notebook AI-ML/
```
or
```bash
uv run jupyter lab AI-ML/
```

---

## 📂 Curriculum Progression

1. **[`00_introduction.ipynb`](00_introduction.ipynb)**: AI and Machine Learning Foundations, Learning Tasks $(T, P, E)$, MLOps, Validation Strategy.
2. **[`01_classifiers-kNN.ipynb`](01_classifiers-kNN.ipynb)**: Instance-Based Classification, Distance Metrics, Feature Scaling, Grid Search, W&B Tracking *(Semester Project Baseline Template)*.
3. **[`02_regression.ipynb`](02_regression.ipynb)**: Continuous Prediction, Parameters $\mathbf{w}$, Design Matrix $\mathbf{X}$, OLS, Residuals, $R^2$, MSE, Regularization (Ridge/Lasso).
4. **[`03_logistic_regression.ipynb`](03_logistic_regression.ipynb)**: Single Artificial Neuron, The Sigmoid Bridge $\sigma(\mathbf{w}^\mathrm{T}\mathbf{x} + w_0)$, Decision Boundaries, Imbalance, ROC-AUC.
5. **[`04_NN.ipynb`](04_NN.ipynb)**: Multi-Layer Perceptrons (MLP), Stacking Linear Layers with Non-Linear Activations, Loss Curves, Scikit-Learn Pipelines.
6. **[`05_ensembles.ipynb`](05_ensembles.ipynb)**: Tree Ensembles: Decision Tree Induction & Pruning, Bagging, Random Forests, Boosting (AdaBoost, Gradient Boosting), W&B Logging.
7. **[`06_SVM.ipynb`](06_SVM.ipynb)**: Geometric Margin Maximization, Support Vectors, Soft Margin $C$, Non-Linear Kernels (RBF).
8. **[`07_unsupervised.ipynb`](07_unsupervised.ipynb)**: Unsupervised Structure Discovery without Labels, $k$-Means, Hierarchical Clustering, PCA.
9. **[`08_ml_agents.ipynb`](08_ml_agents.ipynb)**: AI Coding Agents in Applied ML: Specification-Driven Engineering, $(T, P, E)$ Invariants, TDD, Modular Architectures, W&B Tracking, Defensio Readiness.
10. **[`aux1_BN.ipynb`](aux1_BN.ipynb)** *(Auxiliary)*: Probabilistic Generative Models, Bayes' Theorem, Priors, Conditional Independence, Laplace Smoothing, Text & Categorical Classification.

- **Course Policies & Modalities**: See [`modalities.md`](modalities.md) for grading scale, group project, deadlines, and GenAI policy.

---

## 📽️ Presenting Slides (RISE)

All lecture notebooks are equipped with Reveal.js / RISE slide metadata:
- Press **`Alt + R`** in Jupyter Notebook to start interactive fullscreen slides.
- Viewport heights are strictly calibrated ($\le 820$px) for Full HD 1080p projectors.
- Presentation styling is loaded from `shared/styles/rise.css`.
- Uses native Markdown callouts (`> ⚠️ **Important:**`, `> 💡 **Key Takeaway:**`, `> 📘 **Definition:**`, `<small>📖 *Source: ...*</small>`).

---

## ⚙️ Configuration & Custom Mirrors

By default, notebooks fetch datasets from the official course repository using `helpers.py`:
- `hp.get_file("dataset.csv")` automatically resolves local files or fetches them on standalone Colab runs.
- `hp.get_data_path("")` resolves directory roots for dataset loaders.
- To use a local mirror or private fork, configure `COURSE_DATA_URL`:
  ```bash
  export COURSE_DATA_URL="https://your-mirror-url/shared/data"
  ```
