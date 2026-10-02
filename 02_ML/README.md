# 🤖 AI and Machine Learning Course

Welcome to the **AI and Machine Learning** course module. This course is designed to develop practical analytical skills:
- Formulating real-world learning tasks $(T, P, E)$.
- Selecting appropriate machine learning algorithms based on problem characteristics.
- Rigorously training, tuning, and evaluating models.
- Interpreting results to make informed domain decisions.

---

## 🚀 Running the Notebooks

You can run these notebooks in two ways:
1. **Google Colab (Recommended for zero-install quickstart)**
2. **Local Environment via `uv` (Recommended for offline and project work)**

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
uv run jupyter notebook 02_ML/
```
or
```bash
uv run jupyter lab 02_ML/
```

---

## 📂 Curriculum Progression

1. **`00_introduction.ipynb`**: Machine Learning Foundations, Learning Tasks $(T, P, E)$, MLOps, Validation Strategy.
2. **`01_classifiers-kNN.ipynb`**: Instance-Based Classification, Distance Metrics, Feature Scaling, Grid Search *(Semester Project Baseline Template)*.
3. **`02_regression.ipynb`**: Continuous Prediction, Parameters $\mathbf{w}$, Design Matrix $\mathbf{X}$, OLS, Residuals, $R^2$, MSE, Overfitting.
4. **`03_logistic_regression.ipynb`**: Single Artificial Neuron, The Sigmoid Bridge $\sigma(\mathbf{w}^\mathrm{T}\mathbf{x} + w_0)$, Decision Boundaries, ROC-AUC.
5. **`04_NN.ipynb`**: Multi-Layer Perceptrons (MLP), Stacking Linear Layers with Non-Linear Activations, Scikit-Learn Pipelines.
6. **`05_classifiers-NB.ipynb`**: Probabilistic Generative Models, Bayes' Theorem, Priors, Conditional Independence, Text Classification.
7. **`06_SVM.ipynb`**: Geometric Margin Maximization, Support Vectors, Soft Margin $C$, Non-Linear Kernels (RBF).
8. **`07_unsupervised.ipynb`**: Unsupervised Structure Discovery without Labels, $k$-Means, Hierarchical Clustering, PCA.

---

## ⚙️ Configuration & Custom Mirrors

By default, notebooks fetch datasets from the official course repository. If you are using a local mirror or private fork, you can customize the data base URL by setting the `COURSE_DATA_URL` environment variable:

```bash
export COURSE_DATA_URL="https://your-mirror-url/02_ML/data"
```
