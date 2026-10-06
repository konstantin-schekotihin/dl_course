"""
Self-contained helper utilities for Deep Neural Architectures (ML-DL / 02_deep_learning).
Provides unified visualization and operational routines for neural networks, CNNs,
RNNs, Transformers, Graph Neural Networks (GNNs), and stochastic optimization.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
import torch

try:
    from torch_geometric.utils import to_networkx
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

plt_x, plt_y = 8, 8


# ---------------------------------------------------------------------------
# Binary Classification & Logic Functions (ANN & Deep ANN)
# ---------------------------------------------------------------------------

def plot_binary(predictor, X, y):
    """Plot 2D decision boundary for binary classification models."""
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c='w')

    ax = plt.gca()
    y1, y2 = ax.get_ylim()
    x1, x2 = ax.get_xlim()
    xm, ym = np.meshgrid(
        np.arange(x1, x2, (x2 - x1) / 100),
        np.arange(y1, y2, (y2 - y1) / 100)
    )
    p = predictor(np.c_[xm.ravel(), ym.ravel()]).reshape(xm.shape)
    plt.scatter(xm, ym, c=p, cmap='coolwarm', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='w', s=100, linewidths=2)
    plt.show()


def rescale(X, min=-1, max=1):
    """Rescale tensor or array into [min, max]."""
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std * (max - min) + min


def fn_and(X):
    """Logical AND for 2D inputs."""
    return torch.logical_and(X[:, 0], X[:, 1]).float()


def fn_or(X):
    """Logical OR for 2D inputs."""
    return torch.logical_or(X[:, 0], X[:, 1]).float()


def fn_xor(X):
    """Logical XOR for 2D inputs."""
    return torch.logical_xor(X[:, 0], X[:, 1]).float()


def switch_fn(fn):
    """Return logic function by name."""
    return {'and': fn_and, 'or': fn_or, 'xor': fn_xor}.get(fn)


# ---------------------------------------------------------------------------
# Polymorphic Annotation (CNN Feature Maps & Regression Pairgrids)
# ---------------------------------------------------------------------------

def annotate(first_arg, second_arg, **kwargs):
    """
    Polymorphic annotation helper:
    1. CNN Mode: When called with (im, ax), annotates numeric pixel values onto image patches.
    2. Regression Mode: When called with (x, y, **kws), annotates Pearson correlation onto pairgrid.
    """
    if hasattr(second_arg, 'text') and hasattr(first_arg, 'shape') and len(first_arg.shape) >= 2:
        # CNN Kernel / Feature Map mode: first_arg is im, second_arg is ax
        im = first_arg
        ax = second_arg
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                val = im[i, j].item() if hasattr(im[i, j], 'item') else float(im[i, j])
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", color="r", fontsize=14, weight="bold"
                )
    else:
        # Seaborn PairGrid Pearson correlation mode: first_arg is x, second_arg is y
        x = first_arg
        y = second_arg
        r, p = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate(f"r = {r:.2f}, p = {p:.3f} ", xy=(0.1, 1), xycoords=ax.transAxes)


# ---------------------------------------------------------------------------
# Time Series & Sequence Visualization (RNN / LSTM)
# ---------------------------------------------------------------------------

def plot_series(time, series, format="-", start=0, end=None, label=None):
    """Plot time-series sequences with optional labeling."""
    plt.figure(figsize=(15, 5))
    plt.plot(time[start:end], series[start:end], format, label=label)
    if label:
        plt.legend(loc="upper left")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Graph Neural Network Visualization (PyG / NetworkX)
# ---------------------------------------------------------------------------

def plot_graph(G, classes):
    """Render PyTorch Geometric graph using NetworkX circular layout."""
    if PYG_AVAILABLE:
        G_nx = to_networkx(G, to_undirected=True)
    else:
        G_nx = G
    nx.draw_networkx(
        G_nx, pos=nx.circular_layout(G_nx), with_labels=False, node_color=classes
    )
    plt.axis("off")
    plt.show()


def plot_embedding(t, classes, epoch=None, loss=None):
    """Render 2D graph node embeddings colored by class labels."""
    if hasattr(t, "detach"):
        t_arr = t.detach().cpu().numpy()
    else:
        t_arr = np.asarray(t)
    plt.scatter(t_arr[:, 0], t_arr[:, 1], s=140, c=classes, cmap="tab10", edgecolors="k", alpha=0.85)
    if epoch is not None and loss is not None:
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        plt.xlabel(f"Epoch: {epoch}, Loss: {loss_val:.4f}", fontsize=14)
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Numerical Optimization & Gradient Descent (Bishop Ch 7)
# ---------------------------------------------------------------------------

def plot_out(out):
    """Plot optimization convergence loss on a logarithmic scale."""
    grad_val = out[-1]
    if hasattr(grad_val, "item"):
        grad_val = grad_val.item()
    print(f"Finished optimization in {len(out)} iterations with final metric {grad_val:.4e}")
    plt.figure(figsize=(8, 5))
    plt.plot(out, lw=2, color="royalblue")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss / Gradient Norm")
    plt.grid(True)
    plt.show()


def plot_residuals(predict, X, y):
    """Plot linear/polynomial regression fit line and residual errors."""
    est = predict(X.reshape(-1, 1)).reshape(-1,)
    if hasattr(y, "device") and hasattr(est, "device"):
        rss = torch.sum(torch.pow(y - est, 2)).item()
        tss = torch.sum(torch.pow(y - torch.mean(y), 2)).item()
        mae = torch.sum(torch.abs(y - est)).item() / est.shape[0]
        mse = rss / est.shape[0]
        X_np, y_np, est_np = X.cpu().numpy(), y.cpu().numpy(), est.cpu().numpy()
    else:
        rss = float(np.sum(np.power(y - est, 2)))
        tss = float(np.sum(np.power(y - np.mean(y), 2)))
        mae = float(np.sum(np.abs(y - est)) / len(est))
        mse = rss / len(est)
        X_np, y_np, est_np = np.asarray(X), np.asarray(y), np.asarray(est)

    print(f"TSS = {tss:.3f} - total sum of squares")
    print(f"RSS = {rss:.3f} - residual sum of squares")
    print(f"ESS = TSS - RSS = {tss - rss:.3f} - explained sum of squares")
    print(f"MSE = {mse:.3f} - mean squared error")
    print(f"MAE = {mae:.3f} - mean absolute error")

    X_flat = X_np.ravel()
    y_flat = y_np.ravel()
    est_flat = est_np.ravel()

    l = torch.linspace(float(X_flat.min()), float(X_flat.max()), 1000).reshape(-1, 1)
    plt.figure(figsize=(7, 4.5))
    plt.vlines(X_flat, y_flat, est_flat, colors="royalblue", linestyles="--", alpha=0.4, label="Residuals")
    plt.scatter(X_flat, y_flat, color="royalblue", alpha=0.7, s=25, label="Data")
    plt.scatter(X_flat, est_flat, color="firebrick", alpha=0.7, s=25, label="Fitted")
    pred_line = predict(l.float())
    if hasattr(pred_line, "detach"):
        pred_line = pred_line.detach().cpu().numpy().ravel()
    else:
        pred_line = np.asarray(pred_line).ravel()
    plt.plot(l.numpy().ravel(), pred_line, color="forestgreen", lw=2, label="Model")
    plt.legend()
    plt.xlabel("Input Feature (x)")
    plt.ylabel("Target (t)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def g_idea(f, df, x=1, h=0.05):
    """Illustrate 1D gradient step concept with direction arrow."""
    sp = np.linspace(-0.5, 1, 100)
    plt.figure(figsize=(8, 5))
    plt.plot(sp, f(sp), lw=2)
    x_n = x - h * df(x)
    plt.arrow(
        x, f(x), x_n - x, f(x_n) - f(x),
        head_width=0.04, length_includes_head=True, color="r", linewidth=3
    )
    plt.plot(x, f(x), "ro", markersize=8)
    plt.title(f"Gradient Step: x={x:.2f} -> x_next={x_n:.2f}")
    plt.grid(True)
    plt.show()


def plot_trace(f, trace):
    """Plot optimization parameter trajectory over 1D objective surface."""
    max_bound = max(abs(min(trace)), abs(max(trace))) + 0.5
    domain = torch.arange(-max_bound, max_bound, 0.01)
    f_domain = f(domain)
    if hasattr(f_domain, "detach"):
        f_domain = f_domain.detach().cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.plot(domain.numpy(), f_domain, "b-", lw=2, label="Objective f(x)")
    trace_vals = [f(x) for x in trace]
    if hasattr(trace_vals[0], "item"):
        trace_vals = [tv.item() for tv in trace_vals]
    plt.plot(trace, trace_vals, "-ro", markersize=6, label="Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_grad(f, x, y, x1, y1, angle):
    """Plot 3D loss surface and single gradient step projection."""
    fig = plt.figure(figsize=(plt_x, plt_y))
    sp = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(sp, sp)
    Z = f(X, Y)

    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, angle)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f")

    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=plt.cm.summer, linewidth=0.1)
    ax.scatter([x, x], [y, y], [0, f(x, y)], s=40, c="r")
    ax.scatter([x1, x1], [y1, y1], [f(x1, y1), 0], s=40, c="b")
    ax.plot([x, x1], [y, y1], [f(x, y), f(x1, y1)], color="black", lw=2)
    ax.plot([x, x1], [y, y1], [0, 0], color="black", linestyle="--")
    print(f"initial: [{x:.2f}, {y:.2f}, {f(x, y):.2f}], step: [{x1:.2f}, {y1:.2f}, {f(x1, y1):.2f}]")
    plt.show()


def run_opt(opt, x=-8, y=0, h=0.1, angle=60):
    """Interactive widget callback for standard optimizer step."""
    opt.step(x, y, h)
    opt.plot(x, y, angle)
    opt.update_widgets()


def run_momentum(opt, x=-8, y=0, h=0.1, mu=0.5, angle=60):
    """Interactive widget callback for momentum optimizer step."""
    opt.set_mu(mu)
    run_opt(opt, x, y, h, angle)
