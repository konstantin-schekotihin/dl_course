# Machine Learning & Deep Learning (ML-DL)

**Course Track:** Master's in Computer Science / Artificial Intelligence

Lecture slides and notebooks covering the mathematical foundations of machine learning and the core architectures and training methods of deep learning.

---

## 📖 Textbooks

The course material is based on:
- Christopher M. Bishop & Hugh Bishop (2024). [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/). Springer. This is the primary reference, and the course notation follows it.
- Ian Goodfellow, Yoshua Bengio & Aaron Courville (2016). [*Deep Learning*](https://www.deeplearningbook.org/). MIT Press. It was the course's previous main textbook, and much of the material originates from it.

---

## 📂 Contents

1. [`01_foundations/`](01_foundations/) — a short recap of the mathematical and conceptual foundations, which are usually covered in other courses: notation, the history of AI, probability, information theory, linear algebra, functions used in neural networks, and MLOps.
2. [`02_deep_learning/`](02_deep_learning/) — the core of the course, with most of the exercises: single-layer networks for regression and classification, deep networks, gradient descent, backpropagation, regularization, convolutional networks, sequence models, transformers, and graph neural networks.

Each folder's README gives a one-sentence overview of every notebook.

Course organization, grading, deadlines and project requirements are described in [`modalities.md`](modalities.md).

---

## 🚀 Environment Setup

Locally, all notebooks execute in the repository's unified Python 3.12 environment managed by `uv`:

```bash
# Sync environment dependencies (also installs the course helper package dl-course)
uv sync

# Launch Jupyter Notebook (press Alt + R in a notebook to start the RISE slideshow)
uv run jupyter notebook
```

In Google Colab, open a notebook via **File → Open notebook → GitHub** (`konstantin-schekotihin/dl_course`). Its first cell installs the helper package `dl-course` automatically. See the [main README](../README.md) for VS Code and Google Drive setups.
