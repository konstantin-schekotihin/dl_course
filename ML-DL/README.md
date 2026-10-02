# Machine Learning & Deep Learning (ML-DL)

**Course Track:** Master's in Computer Science / Artificial Intelligence  
**Primary Reference Textbook:** Christopher M. Bishop & Hugh Bishop (2024), *Deep Learning: Foundations and Concepts*, Springer.  
**Repository Path:** `ML-DL/`

---

## 📖 Curriculum Overview

This course provides a mathematically rigorous, concept-first foundation in machine learning and deep learning architectures. All theoretical formulations adhere strictly to the notation and pedagogical doctrine of Christopher M. Bishop and Hugh Bishop (2024).

The curriculum is structured into two sequential modules:

### Part 1: Mathematical & Conceptual Foundations (`01_foundations/`)
Foundations corresponding to Chapters 1–3 and mathematical appendices of Bishop & Bishop (2024):
1. **[`00_notation.ipynb`](01_foundations/00_notation.ipynb)**: Native LaTeX mathematical typography, scalar/vector/matrix conventions, and probabilistic foundations.
2. **[`01_Introduction.ipynb`](01_foundations/01_Introduction.ipynb)**: History of artificial intelligence, symbolic vs. connectionist paradigms, learning tasks, and inductive bias.
3. **[`02_probability.ipynb`](01_foundations/02_probability.ipynb)**: Probability theory, sum and product rules, Bayes' theorem, discrete/continuous densities, maximum likelihood estimation.
4. **[`03_information.ipynb`](01_foundations/03_information.ipynb)**: Information theory, Shannon entropy, cross-entropy loss, Kullback-Leibler (KL) divergence, and mutual information.
5. **[`04_algebra.ipynb`](01_foundations/04_algebra.ipynb)**: Linear algebra, vector spaces, matrix factorizations, singular value decomposition (SVD), eigendecomposition, and Einstein summation (`torch.einsum`).
6. **[`05_functions.ipynb`](01_foundations/05_functions.ipynb)**: Multivariable calculus, gradient vectors, Jacobian & Hessian matrices, Taylor approximations, and convex analysis.
7. **[`06_mlops.ipynb`](01_foundations/06_mlops.ipynb)**: Machine Learning Operations (MLOps), data provenance, experiment reproducibility, and deployment lifecycles.

### Part 2: Deep Neural Architectures (`02_deep_learning/`)
Core deep learning architectures corresponding to Chapters 4–13 of Bishop & Bishop (2024):
1. **[`01_ANN.ipynb`](02_deep_learning/01_ANN.ipynb)**: Single-layer networks, Rosenblatt Perceptron, Adaline, activation functions, and decision surfaces (Bishop Ch 4–5).
2. **[`02_Deep_ANN.ipynb`](02_deep_learning/02_Deep_ANN.ipynb)**: Deep feedforward multilayer perceptrons (MLP), backpropagation mechanics, activation gradients, and TensorBoard diagnostic monitoring (Bishop Ch 6, 8).
3. **[`03-CNNs.ipynb`](02_deep_learning/03-CNNs.ipynb)**: Convolution operations, discrete spatial filtering, receptive fields, pooling layers, and equivariance (Bishop Ch 10).
4. **[`031-CNNs-Architectures.ipynb`](02_deep_learning/031-CNNs-Architectures.ipynb)**: Modern convolutional architectures, residual connections (ResNet), VGG, and batch normalization (Bishop Ch 10).
5. **[`04_RNN.ipynb`](02_deep_learning/04_RNN.ipynb)**: Recurrent neural networks (RNN), exploding/vanishing gradients, Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU).
6. **[`05-Transformers.ipynb`](02_deep_learning/05-Transformers.ipynb)**: Sequence transduction, scaled dot-product attention, multi-head self-attention, positional encodings, and Transformer encoders (BERT) (Bishop Ch 12).
7. **[`06-Graphs.ipynb`](02_deep_learning/06-Graphs.ipynb)**: Graph Neural Networks (GNN), permutation equivariance, message passing frameworks, and graph convolutional networks via PyTorch Geometric (Bishop Ch 13).
8. **[`07_numerical_optimization.ipynb`](02_deep_learning/07_numerical_optimization.ipynb)**: First-order optimization, stochastic gradient descent (SGD), Polyak/Nesterov momentum, RMSprop, Adam, and learning rate schedules (Bishop Ch 7).

---

## 🚀 Environment Setup & Presentation

All notebooks execute in the repository's unified Python 3.12 environment managed by `uv`:

```bash
# Sync environment dependencies
uv sync

# Launch Jupyter Notebook
uv run jupyter notebook
```

### Presentation Slides (RISE)
Every notebook includes pre-configured Reveal.js / RISE metadata:
- Press **`Alt + R`** in Jupyter Notebook to start interactive fullscreen slides.
- Viewports are strictly audited for $\le 820$px height (Full HD 1080p projectors).
- Presentation styling is loaded centrally from `shared/styles/rise.css`.
