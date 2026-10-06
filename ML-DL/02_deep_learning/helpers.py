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

import seaborn as sns

plt_x, plt_y = 8, 8


# ---------------------------------------------------------------------------
# Plotting Aesthetics & Course Theme Configuration
# ---------------------------------------------------------------------------

def setup_theme():
    """Configure unified course-standard Seaborn plotting aesthetics."""
    sns.set_theme(
        style="whitegrid",
        palette="deep",
        rc={
            "figure.figsize": (8, 4.2),
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


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


# ---------------------------------------------------------------------------
# Chapter 4: Single-layer Networks: Regression Visualizations
# ---------------------------------------------------------------------------

def plot_synthetic_regression_data(x_dense, t_dense, x_train, t_train):
    """Plot ground-truth signal and training observations."""
    plt.figure(figsize=(7.5, 3.8))
    plt.plot(x_dense.numpy(), t_dense.numpy(), 'forestgreen', lw=2, label=r"Ground Truth: $\sin(2\pi x)$")
    plt.scatter(x_train.numpy(), t_train.numpy(), facecolors='none', edgecolors='royalblue', s=65, lw=2,
                label=f"Training Observations ($N = {len(x_train)}$)")
    plt.title("Synthetic Regression Dataset: True Signal and Noisy Observations", fontsize=11.5)
    plt.xlabel("Input Feature ($x$)")
    plt.ylabel("Target ($t$)")
    plt.ylim(-1.5, 1.5)
    plt.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    plt.show()


def plot_polynomial_fits(x_dense, t_dense, x_train, t_train, fitted_models, degrees):
    """Plot 2x2 grid of polynomial curve fits for varying degrees M."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.0), sharex=True, sharey=True)
    for ax, M in zip(axes.ravel(), degrees):
        w_fit = fitted_models[M]
        y_dense = torch.matmul(torch.vander(x_dense, N=M + 1, increasing=True), w_fit)
        ax.plot(x_dense.numpy(), t_dense.numpy(), 'forestgreen', lw=1.5, label="True")
        ax.scatter(x_train.numpy(), t_train.numpy(), facecolors='none', edgecolors='royalblue', s=40, lw=1.5)
        ax.plot(x_dense.numpy(), y_dense.numpy(), 'firebrick', lw=2, label=f"$M={M}$")
        ax.set_title(f"Polynomial Degree $M={M}$", fontsize=10.5)
        ax.set_ylim(-1.6, 1.6)
        ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.show()


def plot_dataset_size_remedy(x_dense, t_dense, models_by_N):
    """Plot 1x2 panel illustrating mitigating power of sample size N for flexible model M=9."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, (N_val, x_sc, t_sc, y_dense_sc) in zip(axes, models_by_N):
        ax.plot(x_dense.numpy(), t_dense.numpy(), 'forestgreen', lw=1.5, label="True")
        ax.scatter(x_sc.numpy(), t_sc.numpy(), facecolors='none', edgecolors='royalblue', s=30, alpha=0.8)
        ax.plot(x_dense.numpy(), y_dense_sc.numpy(), 'firebrick', lw=2, label="$M=9$")
        ax.set_title(f"Model $M = 9$ with $N = {N_val}$ Samples", fontsize=10.5)
        ax.set_ylim(-1.6, 1.6)
        ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.show()


def plot_regularized_fits(x_dense, t_dense, x_train, t_train, reg_fits, reg_lambdas):
    """Plot 1x2 panel of regularized polynomial fits for M=9."""
    Phi_dense_10 = torch.vander(x_dense, N=10, increasing=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    titles = [r"$\ln\lambda = -18$ (Optimal Regularization)", r"$\ln\lambda = 0$ (Heavy Damping)"]
    for ax, lam, title in zip(axes, reg_lambdas, titles):
        w_reg = reg_fits[lam]
        y_reg_dense = torch.matmul(Phi_dense_10, w_reg)
        ax.plot(x_dense.numpy(), t_dense.numpy(), 'forestgreen', lw=1.5, label="True")
        ax.scatter(x_train.numpy(), t_train.numpy(), facecolors='none', edgecolors='royalblue', s=45, lw=1.5)
        ax.plot(x_dense.numpy(), y_reg_dense.numpy(), 'firebrick', lw=2, label="Fit")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(-1.6, 1.6)
        ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.show()


def plot_basis_functions(x_eval, centers, s=0.15):
    """Plot 1x3 panel displaying canonical families of basis functions."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for p in range(4):
        axes[0].plot(x_eval.numpy(), (x_eval**p).numpy(), lw=2, label=f"$x^{p}$")
    axes[0].set_title("Polynomial Basis", fontsize=10.5)
    axes[0].legend(fontsize=8.5)

    for mu in centers:
        axes[1].plot(x_eval.numpy(), torch.exp(-0.5 * ((x_eval - mu) / s)**2).numpy(), lw=2)
    axes[1].set_title("Gaussian RBF Basis", fontsize=10.5)

    for mu in centers:
        axes[2].plot(x_eval.numpy(), torch.sigmoid((x_eval - mu) / s).numpy(), lw=2)
    axes[2].set_title("Sigmoidal Basis", fontsize=10.5)

    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylim(-0.1, 1.2)
    plt.tight_layout()
    plt.show()


def plot_streaming_adaline_convergence(X_synth, t_synth, x_line, Phi_line, epoch_snapshots, w_analytic):
    """Plot streaming Adaline regression lines across epochs converging to analytical batch solution."""
    plt.figure(figsize=(8, 3.8))
    plt.scatter(X_synth.numpy(), t_synth.numpy(), facecolors='none', edgecolors='royalblue', s=50, lw=1.5,
                label=f"Streaming Samples ($N={len(X_synth)}$)")

    colors = {'1': 'orange', '3': 'darkviolet', '15': 'crimson'}
    for ep, w_snap in epoch_snapshots[1:]:
        y_snap = Phi_line @ w_snap
        plt.plot(x_line.numpy(), y_snap.numpy(), color=colors[str(ep)], lw=1.8, linestyle='--',
                 label=f"Adaline Epoch {ep}")

    y_exact = Phi_line @ w_analytic
    plt.plot(x_line.numpy(), y_exact.numpy(), 'black', lw=2.5, label=r"Batch OLS $\mathbf{w}_\mathrm{ML}$")
    plt.title("Adaline Online LMS: Progressive Convergence to Batch Least Squares", fontsize=11.5)
    plt.xlabel("Input Feature ($x$)")
    plt.ylabel("Target ($t$)")
    plt.legend(loc="lower right", fontsize=9.0)
    plt.tight_layout()
    plt.show()


def plot_l1_l2_sparsity_geometry():
    """Plot 2D parameter space constraint geometry for L2 (circle) vs L1 (diamond) regularization."""
    w1 = np.linspace(-2.0, 2.5, 300)
    w2 = np.linspace(-2.0, 2.5, 300)
    W1, W2 = np.meshgrid(w1, w2)
    w_opt = np.array([1.4, 1.1])
    E_grid = 1.8 * (W1 - w_opt[0])**2 + 0.9 * (W2 - w_opt[1])**2 - 1.2 * (W1 - w_opt[0]) * (W2 - w_opt[1])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].contour(W1, W2, E_grid, levels=[0.2, 0.6, 1.2, 2.0, 3.2], colors='royalblue', alpha=0.8)
    circle = plt.Circle((0, 0), 1.0, color='crimson', fill=False, lw=2.5, label=r"Constraint: $\Vert\mathbf{w}\Vert_2^2 \leq \eta$")
    axes[0].add_patch(circle)
    axes[0].plot(w_opt[0], w_opt[1], 'ko', markersize=6, label=r"Unconstrained $\mathbf{w}^*$")
    axes[0].plot(0.7, 0.71, 'ro', markersize=6, label="Solution (Off-axis)")
    axes[0].axhline(0, color='gray', lw=0.8, linestyle='--')
    axes[0].axvline(0, color='gray', lw=0.8, linestyle='--')
    axes[0].set_title(r"$L_2$ Regularization (Ridge / Weight Decay)", fontsize=10.5)
    axes[0].set_xlabel(r"$w_1$")
    axes[0].set_ylabel(r"$w_2$")
    axes[0].set_xlim(-1.8, 2.2)
    axes[0].set_ylim(-1.8, 2.2)
    axes[0].set_aspect('equal')
    axes[0].legend(loc='lower left', fontsize=8.5)

    axes[1].contour(W1, W2, E_grid, levels=[0.2, 0.6, 1.2, 2.0, 3.2], colors='royalblue', alpha=0.8)
    diamond = plt.Polygon([[1.1, 0], [0, 1.1], [-1.1, 0], [0, -1.1]], color='crimson', fill=False, lw=2.5,
                          label=r"Constraint: $\Vert\mathbf{w}\Vert_1 \leq \eta$")
    axes[1].add_patch(diamond)
    axes[1].plot(w_opt[0], w_opt[1], 'ko', markersize=6, label=r"Unconstrained $\mathbf{w}^*$")
    axes[1].plot(0.0, 1.1, 'ro', markersize=6, label=r"Solution ($w_1 = 0$, Sparse!)")
    axes[1].axhline(0, color='gray', lw=0.8, linestyle='--')
    axes[1].axvline(0, color='gray', lw=0.8, linestyle='--')
    axes[1].set_title(r"$L_1$ Regularization (Lasso / Sparse)", fontsize=10.5)
    axes[1].set_xlabel(r"$w_1$")
    axes[1].set_ylabel(r"$w_2$")
    axes[1].set_xlim(-1.8, 2.2)
    axes[1].set_ylim(-1.8, 2.2)
    axes[1].set_aspect('equal')
    axes[1].legend(loc='lower left', fontsize=8.5)
    plt.tight_layout()
    plt.show()


def plot_l1_l2_outlier_comparison(x_clean, t_corrupt, x_plot, y_l2_plot, y_l1_plot):
    """Plot comparison of L2 squared loss vs L1 absolute loss on sensor-corrupted data."""
    plt.figure(figsize=(7.5, 3.8))
    plt.scatter(x_clean.numpy(), t_corrupt.numpy(), color='royalblue', s=40, label="Data with Sensor Spikes")
    plt.plot(x_plot.numpy(), y_l2_plot.numpy(), 'firebrick', lw=2, label=r"$L_2$ Squared Loss (Skewed)")
    plt.plot(x_plot.numpy(), y_l1_plot.numpy(), 'forestgreen', lw=2.5, linestyle='--',
             label=r"$L_1$ Absolute Loss (Robust Median)")
    plt.title(r"Decision Theory: Robustness of $L_1$ vs $L_2$ Loss", fontsize=11.5)
    plt.xlabel("Input Feature ($x$)")
    plt.ylabel("Target ($t$)")
    plt.legend(loc="upper left", fontsize=9.0)
    plt.tight_layout()
    plt.show()


def plot_ensemble_fits(x_dense_ev, y_true_ev, ensembles):
    """Render 1x3 panel of model ensembles across High, Optimal, and Low regularization."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    panel_specs = [
        (2.0, r"High Regularization ($\ln\lambda = 2.0$)\nHigh Bias, Low Variance"),
        (-0.5, r"Optimal Balance ($\ln\lambda = -0.5$)\nBalanced Bias & Variance"),
        (-4.0, r"Low Regularization ($\ln\lambda = -4.0$)\nLow Bias, High Variance")
    ]
    for ax, (ln_lam, title) in zip(axes, panel_specs):
        preds_all, mean_pred = ensembles[ln_lam]
        for l in range(20):
            ax.plot(x_dense_ev.numpy(), preds_all[l].numpy(), color='firebrick', alpha=0.3, lw=0.9)
        ax.plot(x_dense_ev.numpy(), mean_pred.numpy(), color='darkred', lw=2.4, label=r"Ensemble Mean $\bar{y}(x)$")
        ax.plot(x_dense_ev.numpy(), y_true_ev.numpy(), color='forestgreen', lw=2.0, linestyle='--', label="Ground Truth")
        ax.set_title(title, fontsize=10.0)
        ax.set_xlabel("$x$")
        ax.set_ylim(-1.6, 1.6)
        if ax == axes[0]:
            ax.set_ylabel("$y(x)$")
            ax.legend(loc="lower left", fontsize=8.0)
    plt.tight_layout()
    plt.show()


def plot_bias_variance_curve(ln_lambdas, bias2_list, var_list, total_bv):
    """Plot canonical U-shaped bias-variance trade-off curve vs ln(lambda)."""
    plt.figure(figsize=(7.5, 3.8))
    plt.plot(ln_lambdas, bias2_list, 'royalblue', lw=2.5, label=r"$(\mathrm{Bias})^2$")
    plt.plot(ln_lambdas, var_list, 'firebrick', lw=2.5, label="Variance")
    plt.plot(ln_lambdas, total_bv, 'purple', lw=2.5, linestyle='--', label=r"$(\mathrm{Bias})^2 + \mathrm{Variance}$")
    plt.title(r"The Bias–Variance Trade-Off vs $\ln\lambda$", fontsize=11.5)
    plt.xlabel(r"$\ln\lambda$")
    plt.ylabel("Error")
    plt.ylim(0, 0.12)
    plt.legend(loc="upper center", fontsize=9.0)
    plt.tight_layout()
    plt.show()
