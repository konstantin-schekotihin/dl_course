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
- The first cell installs the course helper package `dl-course`, and datasets are downloaded automatically on first use. No manual file uploads or Drive mounting are required.

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
Alternatively, open the notebooks in VS Code and select the `.venv` interpreter as the kernel (see the [main README](../README.md)).

---

## 📂 Notebooks

| Notebook | Overview |
|---|---|
| [`00_introduction.ipynb`](00_introduction.ipynb) | Introduces AI and machine learning: the three learning paradigms, the formal $(T, P, E)$ learning task, MLOps, and the training and validation workflow. |
| [`01_classifiers-kNN.ipynb`](01_classifiers-kNN.ipynb) | Classifies flowers with $k$-nearest neighbors, from exploratory analysis to cross-validated hyperparameter search and experiment tracking with W&B; it also serves as the semester project template. |
| [`02_regression.ipynb`](02_regression.ipynb) | Linear regression: simple and multiple regression, statistical inference and goodness of fit, polynomial extensions, feature selection and regularization. |
| [`03_logistic_regression.ipynb`](03_logistic_regression.ipynb) | Logistic regression for qualitative targets: the sigmoid model, maximum likelihood with cross-entropy, and a credit-default case study. |
| [`04_NN.ipynb`](04_NN.ipynb) | Neural networks from the perceptron to multi-layer perceptrons, including backpropagation and practical training with scikit-learn. |
| [`05_ensembles.ipynb`](05_ensembles.ipynb) | Decision trees and tree ensembles (bagging, random forests, boosting), applied to breast-cancer diagnosis with W&B tracking. |
| [`06_SVM.ipynb`](06_SVM.ipynb) | Support vector machines: maximal- and soft-margin classifiers and the kernel trick, applied to predicting breast-cancer recurrence. |
| [`07_unsupervised.ipynb`](07_unsupervised.ipynb) | Unsupervised learning with principal component analysis and clustering. |
| [`08_ml_agents.ipynb`](08_ml_agents.ipynb) | A specification-driven workflow for ML projects with AI coding agents, from task specification and baselines to test-driven development and preparation for the oral defense. |
| [`aux1_BN.ipynb`](aux1_BN.ipynb) *(supplementary)* | Bayesian classifiers: Bayes' theorem, ML and MAP estimation, the Bayes optimal classifier, and naive Bayes. |

Course policies (grading, team project, deadlines, GenAI policy) are described in [`modalities.md`](modalities.md). The [project guide](deliverables/) explains how to prepare the project deliverables and provides LaTeX templates for the proposal and the report.

---

## ⚙️ Configuration & Custom Mirrors

By default, notebooks fetch datasets from the official course repository using the course helper package [`dl-course`](../packages/dl-course) (imported as `hp`):
- `hp.get_file("dataset.csv")` automatically resolves local files or fetches them on standalone Colab runs.
- `hp.get_data_path("")` resolves directory roots for dataset loaders.
- To use a mirror or private fork, set `COURSE_RAW_URL` to the URL of its `shared` folder before starting Jupyter. Files are then fetched from `<COURSE_RAW_URL>/data/` and `<COURSE_RAW_URL>/images/`:
  ```bash
  export COURSE_RAW_URL="https://your-mirror-url/shared"
  ```
