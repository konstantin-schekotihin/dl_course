# Machine Learning & Deep Learning (ML-DL)

**Course Track:** Master's in Computer Science / Artificial Intelligence  
**Primary Reference Textbook:** Christopher M. Bishop & Hugh Bishop (2024), *Deep Learning: Foundations and Concepts*, Springer.

---

## 📖 Curriculum Overview

This course provides a foundation in machine learning and deep learning architectures and mostly based on Bishop & Bishop (2024).

The curriculum is structured into two sequential modules:

### Part 1: Mathematical & Conceptual Foundations (`01_foundations/`)
Foundations corresponding to Chapters 1–3 and mathematical appendices of Bishop & Bishop (2024):
1. **[`00_notation.ipynb`](01_foundations/00_notation.ipynb)**: Native LaTeX mathematical typography, scalar/vector/matrix conventions, and Bishop probability notation.
2. **[`01_Introduction.ipynb`](01_foundations/01_Introduction.ipynb)**: History of artificial intelligence, symbolic vs. connectionist paradigms, learning tasks, and inductive bias.
3. **[`02_probability.ipynb`](01_foundations/02_probability.ipynb)**: Probability theory, sum and product rules, Bayes' theorem, discrete/continuous densities, maximum likelihood estimation, and bias-variance decomposition.
4. **[`03_information.ipynb`](01_foundations/03_information.ipynb)**: Information theory, Shannon entropy, cross-entropy loss, Kullback-Leibler (KL) divergence, and mutual information.
5. **[`04_algebra.ipynb`](01_foundations/04_algebra.ipynb)**: Linear algebra, vector spaces, matrix factorizations, singular value decomposition (SVD), eigendecomposition, and Einstein summation (`torch.einsum`).
6. **[`05_functions.ipynb`](01_foundations/05_functions.ipynb)**: Multivariable calculus, gradient vectors, Jacobian & Hessian matrices, Taylor approximations, activation functions, and canonical link theorem.
7. **[`06_mlops.ipynb`](01_foundations/06_mlops.ipynb)**: Machine Learning Operations (MLOps), data provenance, experiment reproducibility, and deployment lifecycles.

### Part 2: Deep Neural Architectures (`02_deep_learning/`)
Core deep learning architectures corresponding to Chapters 4–13 of Bishop & Bishop (2024):
1. **[`04_regression.ipynb`](02_deep_learning/04_regression.ipynb)**: Single-layer Networks: Regression (Bishop Ch. 4 — Linear basis function models, OLS normal equations, regularized least squares, sequential Bayesian updates).
2. **[`05_classification.ipynb`](02_deep_learning/05_classification.ipynb)**: Single-layer Networks: Classification (Bishop Ch. 5 — Discriminant functions, Fisher's linear discriminant, Perceptron, logistic sigmoid, multiclass softmax).
3. **[`06_deep_networks.ipynb`](02_deep_learning/06_deep_networks.ipynb)**: Deep Neural Networks (Bishop Ch. 6 — Multilayer perceptrons, universal approximation theorem, activation functions, representations, FashionMNIST benchmarking).
4. **[`07_gradient_descent.ipynb`](02_deep_learning/07_gradient_descent.ipynb)**: Gradient Descent (Bishop Ch. 7 — Optimization landscapes, batch vs. mini-batch SGD, Polyak/Nesterov momentum, RMSprop, Adam, learning rate schedules).
5. **[`08_backpropagation.ipynb`](02_deep_learning/08_backpropagation.ipynb)**: Backpropagation (Bishop Ch. 8 — Computational graphs, reverse-mode automatic differentiation, vector chain rule, pure NumPy backprop verified against PyTorch Autograd).
6. **[`09_regularization.ipynb`](02_deep_learning/09_regularization.ipynb)**: Regularization (Bishop Ch. 9 — $L_2$ weight decay, early stopping, data augmentation, dropout, batch normalization).
7. **[`10_convolutional_networks.ipynb`](02_deep_learning/10_convolutional_networks.ipynb)**: Convolutional Networks (Bishop Ch. 10 — Spatial filtering, weight sharing, translational equivariance, receptive fields, modern CNN architectures).
8. **[`11_sequence_models.ipynb`](02_deep_learning/11_sequence_models.ipynb)**: Sequence Models (Bishop Ch. 11 — Recurrent neural networks, exploding/vanishing gradients, LSTM, GRU, sequence forecasting).
9. **[`12_transformers.ipynb`](02_deep_learning/12_transformers.ipynb)**: Transformers (Bishop Ch. 12 — Sequence transduction, scaled dot-product attention, multi-head self-attention, positional encodings, encoder-decoder architectures).
10. **[`13_graph_neural_networks.ipynb`](02_deep_learning/13_graph_neural_networks.ipynb)**: Graph Neural Networks (Bishop Ch. 13 — Permutation equivariance, message passing frameworks, GCN via PyTorch Geometric).

---

## 🚀 Environment Setup

All notebooks execute in the repository's unified Python 3.12 environment managed by `uv`:

```bash
# Sync environment dependencies
uv sync

# Launch Jupyter Notebook
uv run jupyter notebook
```