"""
Unified and modular helper utilities for Deep Neural Architectures (ML-DL / 02_deep_learning).
Provides structured, decoupled visualization and operational routines for:
1. Course Aesthetics & Theme Configuration
2. Regression, Curve Fitting & Basis Expansions (04_regression)
3. Classification & Decision Boundaries (05_classification, 06_deep_networks)
4. Optimization Landscapes & Dynamics (07_gradient_descent, 08_backpropagation)
5. Deep Architecture Diagnostics & Convolutions (09_regularization, 10_convolutional_networks)
6. Sequences, Attention & Relational Graphs (11_sequence_models, 12_transformers, 13_graph_neural_networks)
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

try:
    from torch_geometric.utils import to_networkx
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

plt_x, plt_y = 8, 8


# ---------------------------------------------------------------------------
# 1. Course Aesthetics & Theme Configuration
# ---------------------------------------------------------------------------

def setup_theme() -> None:
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
        },
    )


# ---------------------------------------------------------------------------
# 2. Regression, Curve Fitting & Basis Expansions (04_regression)
# ---------------------------------------------------------------------------

def plot_regression_fit(
    x_train: Any,
    t_train: Any,
    x_eval: Any,
    y_eval: Any,
    y_true: Optional[Any] = None,
    residuals: bool = False,
    title: Optional[str] = None,
    xlabel: str = "Input Feature ($x$)",
    ylabel: str = "Target ($t$)",
    ylim: Optional[Tuple[float, float]] = (-1.5, 1.5),
    figsize: Tuple[float, float] = (7.5, 3.8),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Decoupled 1D regression curve visualizer.
    Renders observations, fitted model curve, optional ground-truth, and residual drop-lines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x_tr_np = x_train.numpy() if hasattr(x_train, "numpy") else np.asarray(x_train)
    t_tr_np = t_train.numpy() if hasattr(t_train, "numpy") else np.asarray(t_train)
    x_ev_np = x_eval.numpy() if hasattr(x_eval, "numpy") else np.asarray(x_eval)
    y_ev_np = y_eval.numpy() if hasattr(y_eval, "numpy") else np.asarray(y_eval)

    if y_true is not None:
        y_tr_true = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
        ax.plot(x_ev_np.ravel(), y_tr_true.ravel(), color="forestgreen", lw=2, label="Ground Truth")

    ax.scatter(
        x_tr_np.ravel(),
        t_tr_np.ravel(),
        facecolors="none",
        edgecolors="royalblue",
        s=55,
        lw=1.8,
        label=f"Observations ($N={len(x_tr_np)}$)",
    )
    ax.plot(x_ev_np.ravel(), y_ev_np.ravel(), color="firebrick", lw=2.2, label="Model Prediction")

    if residuals:
        ax.vlines(
            x_tr_np.ravel(),
            t_tr_np.ravel(),
            np.interp(x_tr_np.ravel(), x_ev_np.ravel(), y_ev_np.ravel()),
            colors="royalblue",
            linestyles="--",
            alpha=0.5,
            label="Residuals",
        )

    if title:
        ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", fontsize=9.5)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_synthetic_regression_data(
    x_dense: Any,
    t_dense: Any,
    x_train: Any,
    t_train: Any,
    title: str = "Synthetic Regression Dataset: True Signal and Noisy Observations",
    figsize: Tuple[float, float] = (7.5, 3.8),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ground-truth signal and training observations for Chapter 4."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x_d = x_dense.numpy() if hasattr(x_dense, "numpy") else np.asarray(x_dense)
    t_d = t_dense.numpy() if hasattr(t_dense, "numpy") else np.asarray(t_dense)
    x_tr = x_train.numpy() if hasattr(x_train, "numpy") else np.asarray(x_train)
    t_tr = t_train.numpy() if hasattr(t_train, "numpy") else np.asarray(t_train)

    ax.plot(x_d.ravel(), t_d.ravel(), "forestgreen", lw=2, label=r"Ground Truth: $\sin(2\pi x)$")
    ax.scatter(
        x_tr.ravel(),
        t_tr.ravel(),
        facecolors="none",
        edgecolors="royalblue",
        s=65,
        lw=2,
        label=f"Training Observations ($N = {len(x_tr)}$)",
    )
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Input Feature ($x$)", fontsize=11)
    ax.set_ylabel("Target ($t$)", fontsize=11)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_polynomial_fits(
    x_dense: Any,
    t_dense: Any,
    x_train: Any,
    t_train: Any,
    fitted_models: Dict[int, Any],
    degrees: Sequence[int],
    figsize: Tuple[float, float] = (9.5, 5.0),
) -> None:
    """Plot 2x2 grid of polynomial curve fits for varying degrees M."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    x_d = x_dense.numpy() if hasattr(x_dense, "numpy") else np.asarray(x_dense)
    t_d = t_dense.numpy() if hasattr(t_dense, "numpy") else np.asarray(t_dense)
    x_tr = x_train.numpy() if hasattr(x_train, "numpy") else np.asarray(x_train)
    t_tr = t_train.numpy() if hasattr(t_train, "numpy") else np.asarray(t_train)

    for ax, M in zip(axes.ravel(), degrees):
        w_fit = fitted_models[M]
        y_dense = torch.matmul(torch.vander(x_dense, N=M + 1, increasing=True), w_fit)
        y_d = y_dense.numpy() if hasattr(y_dense, "numpy") else np.asarray(y_dense)

        ax.plot(x_d.ravel(), t_d.ravel(), "forestgreen", lw=1.5, label="True")
        ax.scatter(x_tr.ravel(), t_tr.ravel(), facecolors="none", edgecolors="royalblue", s=40, lw=1.5)
        ax.plot(x_d.ravel(), y_d.ravel(), "firebrick", lw=2, label=f"$M={M}$")
        ax.set_title(f"Polynomial Degree $M={M}$", fontsize=10.5)
        ax.set_ylim(-1.6, 1.6)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_dataset_size_remedy(
    x_dense: Any,
    t_dense: Any,
    models_by_N: Sequence[Tuple[int, Any, Any, Any]],
    figsize: Tuple[float, float] = (9.5, 3.8),
) -> None:
    """Plot 1x2 panel illustrating mitigating power of sample size N for flexible model M=9."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    x_d = x_dense.numpy() if hasattr(x_dense, "numpy") else np.asarray(x_dense)
    t_d = t_dense.numpy() if hasattr(t_dense, "numpy") else np.asarray(t_dense)

    for ax, (N_val, x_sc, t_sc, y_dense_sc) in zip(axes, models_by_N):
        x_s = x_sc.numpy() if hasattr(x_sc, "numpy") else np.asarray(x_sc)
        t_s = t_sc.numpy() if hasattr(t_sc, "numpy") else np.asarray(t_sc)
        y_s = y_dense_sc.numpy() if hasattr(y_dense_sc, "numpy") else np.asarray(y_dense_sc)

        ax.plot(x_d.ravel(), t_d.ravel(), "forestgreen", lw=1.5, label="True")
        ax.scatter(x_s.ravel(), t_s.ravel(), facecolors="none", edgecolors="royalblue", s=30, alpha=0.8)
        ax.plot(x_d.ravel(), y_s.ravel(), "firebrick", lw=2, label="$M=9$")
        ax.set_title(f"Model $M = 9$ with $N = {N_val}$ Samples", fontsize=10.5)
        ax.set_ylim(-1.6, 1.6)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_regularized_fits(
    x_dense: Any,
    t_dense: Any,
    x_train: Any,
    t_train: Any,
    reg_fits: Dict[Any, Any],
    reg_lambdas: Sequence[Any],
    figsize: Tuple[float, float] = (9.5, 3.8),
) -> None:
    """Plot 1x2 panel of regularized polynomial fits for M=9."""
    Phi_dense_10 = torch.vander(x_dense, N=10, increasing=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    titles = [r"$\ln\lambda = -18$ (Optimal Regularization)", r"$\ln\lambda = 0$ (Heavy Damping)"]
    x_d = x_dense.numpy() if hasattr(x_dense, "numpy") else np.asarray(x_dense)
    t_d = t_dense.numpy() if hasattr(t_dense, "numpy") else np.asarray(t_dense)
    x_tr = x_train.numpy() if hasattr(x_train, "numpy") else np.asarray(x_train)
    t_tr = t_train.numpy() if hasattr(t_train, "numpy") else np.asarray(t_train)

    for ax, lam, title in zip(axes, reg_lambdas, titles):
        w_reg = reg_fits[lam]
        y_reg_dense = torch.matmul(Phi_dense_10, w_reg)
        y_reg = y_reg_dense.numpy() if hasattr(y_reg_dense, "numpy") else np.asarray(y_reg_dense)

        ax.plot(x_d.ravel(), t_d.ravel(), "forestgreen", lw=1.5, label="True")
        ax.scatter(x_tr.ravel(), t_tr.ravel(), facecolors="none", edgecolors="royalblue", s=45, lw=1.5)
        ax.plot(x_d.ravel(), y_reg.ravel(), "firebrick", lw=2, label="Fit")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(-1.6, 1.6)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_basis_functions(
    x_eval: Any,
    centers: Sequence[float],
    s: float = 0.15,
    figsize: Tuple[float, float] = (10.5, 3.4),
) -> None:
    """Plot 1x3 panel displaying canonical families of basis functions."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x_ev = x_eval.numpy() if hasattr(x_eval, "numpy") else np.asarray(x_eval)

    for p in range(4):
        axes[0].plot(x_ev, x_ev ** p, lw=2, label=f"$x^{p}$")
    axes[0].set_title("Polynomial Basis", fontsize=10.5)
    axes[0].legend(fontsize=8.5)

    for mu in centers:
        mu_val = mu.item() if hasattr(mu, "item") else float(mu)
        axes[1].plot(x_ev, np.exp(-0.5 * ((x_ev - mu_val) / s) ** 2), lw=2)
    axes[1].set_title("Gaussian RBF Basis", fontsize=10.5)

    for mu in centers:
        mu_val = mu.item() if hasattr(mu, "item") else float(mu)
        axes[2].plot(x_ev, 1.0 / (1.0 + np.exp(-(x_ev - mu_val) / s)), lw=2)
    axes[2].set_title("Sigmoidal Basis", fontsize=10.5)

    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_streaming_adaline_convergence(
    X_synth: Any,
    t_synth: Any,
    x_line: Any,
    Phi_line: Any,
    epoch_snapshots: Sequence[Tuple[int, Any]],
    w_analytic: Any,
    figsize: Tuple[float, float] = (8, 3.8),
) -> None:
    """Plot streaming Adaline regression lines across epochs converging to analytical batch solution."""
    plt.figure(figsize=figsize)
    X_s = X_synth.numpy() if hasattr(X_synth, "numpy") else np.asarray(X_synth)
    t_s = t_synth.numpy() if hasattr(t_synth, "numpy") else np.asarray(t_synth)
    x_l = x_line.numpy() if hasattr(x_line, "numpy") else np.asarray(x_line)

    plt.scatter(
        X_s.ravel(),
        t_s.ravel(),
        facecolors="none",
        edgecolors="royalblue",
        s=50,
        lw=1.5,
        label=f"Streaming Samples ($N={len(X_s)}$)",
    )

    colors = {"1": "orange", "3": "darkviolet", "15": "crimson"}
    for ep, w_snap in epoch_snapshots[1:]:
        y_snap = Phi_line @ w_snap
        y_sn = y_snap.numpy() if hasattr(y_snap, "numpy") else np.asarray(y_snap)
        plt.plot(
            x_l.ravel(),
            y_sn.ravel(),
            color=colors.get(str(ep), "gray"),
            lw=1.8,
            linestyle="--",
            label=f"Adaline Epoch {ep}",
        )

    y_exact = Phi_line @ w_analytic
    y_ex = y_exact.numpy() if hasattr(y_exact, "numpy") else np.asarray(y_exact)
    plt.plot(x_l.ravel(), y_ex.ravel(), "black", lw=2.5, label=r"Batch OLS $\mathbf{w}_\mathrm{ML}$")

    plt.title("Adaline Online LMS: Progressive Convergence to Batch Least Squares", fontsize=11.5)
    plt.xlabel("Input Feature ($x$)")
    plt.ylabel("Target ($t$)")
    plt.legend(loc="lower right", fontsize=9.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_l1_l2_sparsity_geometry(figsize: Tuple[float, float] = (9.5, 3.8)) -> None:
    """Plot 2D parameter space constraint geometry for L2 (circle) vs L1 (diamond) regularization."""
    w1 = np.linspace(-2.0, 2.5, 300)
    w2 = np.linspace(-2.0, 2.5, 300)
    W1, W2 = np.meshgrid(w1, w2)
    w_opt = np.array([1.4, 1.1])
    E_grid = 1.8 * (W1 - w_opt[0]) ** 2 + 0.9 * (W2 - w_opt[1]) ** 2 - 1.2 * (W1 - w_opt[0]) * (W2 - w_opt[1])

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].contour(W1, W2, E_grid, levels=[0.2, 0.6, 1.2, 2.0, 3.2], colors="royalblue", alpha=0.8)
    circle = plt.Circle((0, 0), 1.0, color="crimson", fill=False, lw=2.5, label=r"Constraint: $\Vert\mathbf{w}\Vert_2^2 \leq \eta$")
    axes[0].add_patch(circle)
    axes[0].plot(w_opt[0], w_opt[1], "ko", markersize=6, label=r"Unconstrained $\mathbf{w}^*$")
    axes[0].plot(0.7, 0.71, "ro", markersize=6, label="Solution (Off-axis)")
    axes[0].axhline(0, color="gray", lw=0.8, linestyle="--")
    axes[0].axvline(0, color="gray", lw=0.8, linestyle="--")
    axes[0].set_title(r"$L_2$ Regularization (Ridge / Weight Decay)", fontsize=10.5)
    axes[0].set_xlabel(r"$w_1$")
    axes[0].set_ylabel(r"$w_2$")
    axes[0].set_xlim(-1.8, 2.2)
    axes[0].set_ylim(-1.8, 2.2)
    axes[0].set_aspect("equal")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].grid(True)

    axes[1].contour(W1, W2, E_grid, levels=[0.2, 0.6, 1.2, 2.0, 3.2], colors="royalblue", alpha=0.8)
    diamond = plt.Polygon(
        [[1.1, 0], [0, 1.1], [-1.1, 0], [0, -1.1]],
        color="crimson",
        fill=False,
        lw=2.5,
        label=r"Constraint: $\Vert\mathbf{w}\Vert_1 \leq \eta$",
    )
    axes[1].add_patch(diamond)
    axes[1].plot(w_opt[0], w_opt[1], "ko", markersize=6, label=r"Unconstrained $\mathbf{w}^*$")
    axes[1].plot(0.0, 1.1, "ro", markersize=6, label=r"Solution ($w_1 = 0$, Sparse!)")
    axes[1].axhline(0, color="gray", lw=0.8, linestyle="--")
    axes[1].axvline(0, color="gray", lw=0.8, linestyle="--")
    axes[1].set_title(r"$L_1$ Regularization (Lasso / Sparse)", fontsize=10.5)
    axes[1].set_xlabel(r"$w_1$")
    axes[1].set_ylabel(r"$w_2$")
    axes[1].set_xlim(-1.8, 2.2)
    axes[1].set_ylim(-1.8, 2.2)
    axes[1].set_aspect("equal")
    axes[1].legend(loc="lower left", fontsize=8.5)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_l1_l2_outlier_comparison(
    x_clean: Any,
    t_corrupt: Any,
    x_plot: Any,
    y_l2_plot: Any,
    y_l1_plot: Any,
    figsize: Tuple[float, float] = (7.5, 3.8),
) -> None:
    """Plot comparison of L2 squared loss vs L1 absolute loss on sensor-corrupted data."""
    plt.figure(figsize=figsize)
    x_c = x_clean.numpy() if hasattr(x_clean, "numpy") else np.asarray(x_clean)
    t_c = t_corrupt.numpy() if hasattr(t_corrupt, "numpy") else np.asarray(t_corrupt)
    x_p = x_plot.numpy() if hasattr(x_plot, "numpy") else np.asarray(x_plot)
    y_l2 = y_l2_plot.numpy() if hasattr(y_l2_plot, "numpy") else np.asarray(y_l2_plot)
    y_l1 = y_l1_plot.numpy() if hasattr(y_l1_plot, "numpy") else np.asarray(y_l1_plot)

    plt.scatter(x_c.ravel(), t_c.ravel(), color="royalblue", s=40, label="Data with Sensor Spikes")
    plt.plot(x_p.ravel(), y_l2.ravel(), "firebrick", lw=2, label=r"$L_2$ Squared Loss (Skewed)")
    plt.plot(
        x_p.ravel(),
        y_l1.ravel(),
        "forestgreen",
        lw=2.5,
        linestyle="--",
        label=r"$L_1$ Absolute Loss (Robust Median)",
    )
    plt.title(r"Decision Theory: Robustness of $L_1$ vs $L_2$ Loss", fontsize=11.5)
    plt.xlabel("Input Feature ($x$)")
    plt.ylabel("Target ($t$)")
    plt.legend(loc="upper left", fontsize=9.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ensemble_fits(
    x_dense_ev: Any,
    y_true_ev: Any,
    ensembles: Dict[float, Tuple[Any, Any]],
    figsize: Tuple[float, float] = (11, 3.8),
) -> None:
    """Render 1x3 panel of model ensembles across High, Optimal, and Low regularization."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    panel_specs = [
        (2.0, r"High Regularization ($\ln\lambda = 2.0$)\nHigh Bias, Low Variance"),
        (-0.5, r"Optimal Balance ($\ln\lambda = -0.5$)\nBalanced Bias & Variance"),
        (-4.0, r"Low Regularization ($\ln\lambda = -4.0$)\nLow Bias, High Variance"),
    ]
    x_ev = x_dense_ev.numpy() if hasattr(x_dense_ev, "numpy") else np.asarray(x_dense_ev)
    y_tr = y_true_ev.numpy() if hasattr(y_true_ev, "numpy") else np.asarray(y_true_ev)

    for ax, (ln_lam, title) in zip(axes, panel_specs):
        preds_all, mean_pred = ensembles[ln_lam]
        mean_p = mean_pred.numpy() if hasattr(mean_pred, "numpy") else np.asarray(mean_pred)

        for l in range(min(20, len(preds_all))):
            pred_l = preds_all[l].numpy() if hasattr(preds_all[l], "numpy") else np.asarray(preds_all[l])
            ax.plot(x_ev.ravel(), pred_l.ravel(), color="firebrick", alpha=0.3, lw=0.9)

        ax.plot(x_ev.ravel(), mean_p.ravel(), color="darkred", lw=2.4, label=r"Ensemble Mean $\bar{y}(x)$")
        ax.plot(x_ev.ravel(), y_tr.ravel(), color="forestgreen", lw=2.0, linestyle="--", label="Ground Truth")
        ax.set_title(title, fontsize=10.0)
        ax.set_xlabel("$x$")
        ax.set_ylim(-1.6, 1.6)
        ax.grid(True)
        if ax == axes[0]:
            ax.set_ylabel("$y(x)$")
            ax.legend(loc="lower left", fontsize=8.0)

    plt.tight_layout()
    plt.show()


def plot_bias_variance_curve(
    ln_lambdas: Sequence[float],
    bias2_list: Sequence[float],
    var_list: Sequence[float],
    total_bv: Sequence[float],
    xlabel: str = r"$\ln\lambda$",
    ylabel: str = "Error",
    ylim: Tuple[float, float] = (0, 0.12),
    figsize: Tuple[float, float] = (7.5, 3.8),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot canonical U-shaped bias-variance trade-off curve vs ln(lambda)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(ln_lambdas, bias2_list, "royalblue", lw=2.5, label=r"$(\mathrm{Bias})^2$")
    ax.plot(ln_lambdas, var_list, "firebrick", lw=2.5, label="Variance")
    ax.plot(ln_lambdas, total_bv, "purple", lw=2.5, linestyle="--", label=r"$(\mathrm{Bias})^2 + \mathrm{Variance}$")
    ax.set_title(r"The Bias–Variance Trade-Off vs $\ln\lambda$", fontsize=11.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(ylim)
    ax.legend(loc="upper center", fontsize=9.0)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_residuals(
    predict: Callable[[Any], Any],
    X: Any,
    y: Any,
    figsize: Tuple[float, float] = (7, 4.5),
) -> None:
    """Plot linear/polynomial regression fit line and residual errors (backward-compatible)."""
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
    plt.figure(figsize=figsize)
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


# ---------------------------------------------------------------------------
# 3. Classification & Decision Boundaries (05_classification, 06_deep_networks)
# ---------------------------------------------------------------------------

def plot_decision_boundary_2d(
    predictor: Callable[[np.ndarray], np.ndarray],
    X: Any,
    y: Any,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D decision boundary meshgrid and data points for binary classification."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    X_np = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
    y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)

    x1_min, x1_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    x2_min, x2_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5

    xm, ym = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200),
    )
    grid_pts = np.c_[xm.ravel(), ym.ravel()]
    p = predictor(grid_pts).reshape(xm.shape)

    ax.contourf(xm, ym, p, alpha=0.3, cmap="coolwarm")
    scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np.ravel(), cmap="coolwarm", edgecolors="k", s=80, lw=1.5)
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel("$x_1$", fontsize=11.5)
    ax.set_ylabel("$x_2$", fontsize=11.5)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_binary(predictor: Callable[[np.ndarray], np.ndarray], X: Any, y: Any) -> None:
    """Plot 2D decision boundary for binary classification models (backward-compatible)."""
    plot_decision_boundary_2d(predictor, X, y)


def rescale(X: Any, min: float = -1.0, max: float = 1.0) -> Any:
    """Rescale tensor or array into [min, max]."""
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std * (max - min) + min


def fn_and(X: Any) -> Any:
    """Logical AND for 2D inputs."""
    return torch.logical_and(X[:, 0], X[:, 1]).float()


def fn_or(X: Any) -> Any:
    """Logical OR for 2D inputs."""
    return torch.logical_or(X[:, 0], X[:, 1]).float()


def fn_xor(X: Any) -> Any:
    """Logical XOR for 2D inputs."""
    return torch.logical_xor(X[:, 0], X[:, 1]).float()


def switch_fn(fn: str) -> Optional[Callable[[Any], Any]]:
    """Return logic function by name."""
    return {"and": fn_and, "or": fn_or, "xor": fn_xor}.get(fn)


# ---------------------------------------------------------------------------
# 4. Optimization Landscapes & Dynamics (07_gradient_descent, 08_backpropagation)
# ---------------------------------------------------------------------------

def plot_loss_history(
    losses: Sequence[float],
    log_scale: bool = True,
    title: str = "Optimization Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Loss / Metric",
    figsize: Tuple[float, float] = (8, 4.5),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot optimization convergence loss with optional logarithmic scaling."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(losses, lw=2.2, color="royalblue")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_out(out: Sequence[Any]) -> None:
    """Plot optimization convergence loss on a logarithmic scale (backward-compatible)."""
    grad_val = out[-1]
    if hasattr(grad_val, "item"):
        grad_val = grad_val.item()
    print(f"Finished optimization in {len(out)} iterations with final metric {grad_val:.4e}")
    plot_loss_history(out, log_scale=True)


def g_idea(f: Callable[[Any], Any], df: Callable[[Any], Any], x: float = 1.0, h: float = 0.05) -> None:
    """Illustrate 1D gradient step concept with direction arrow."""
    sp = np.linspace(-0.5, 1, 100)
    plt.figure(figsize=(8, 5))
    plt.plot(sp, f(sp), lw=2, color="royalblue")
    x_n = x - h * df(x)
    plt.arrow(
        x,
        f(x),
        x_n - x,
        f(x_n) - f(x),
        head_width=0.04,
        length_includes_head=True,
        color="crimson",
        linewidth=3,
    )
    plt.plot(x, f(x), "ro", markersize=8)
    plt.title(f"Gradient Step: x={x:.2f} -> x_next={x_n:.2f}", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trace(f: Callable[[Any], Any], trace: Sequence[Any]) -> None:
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
    plt.tight_layout()
    plt.show()


def plot_grad(
    f: Callable[[Any, Any], Any],
    x: float,
    y: float,
    x1: float,
    y1: float,
    angle: float = 60.0,
) -> None:
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
    plt.tight_layout()
    plt.show()


def run_opt(opt: Any, x: float = -8.0, y: float = 0.0, h: float = 0.1, angle: float = 60.0) -> None:
    """Interactive widget callback for standard optimizer step."""
    opt.step(x, y, h)
    opt.plot(x, y, angle)
    opt.update_widgets()


def run_momentum(
    opt: Any,
    x: float = -8.0,
    y: float = 0.0,
    h: float = 0.1,
    mu: float = 0.5,
    angle: float = 60.0,
) -> None:
    """Interactive widget callback for momentum optimizer step."""
    opt.set_mu(mu)
    run_opt(opt, x, y, h, angle)


# ---------------------------------------------------------------------------
# 5. Deep Architecture Diagnostics & Convolutions (09_reg, 10_cnn)
# ---------------------------------------------------------------------------

def plot_activation_curves(
    act_name: str = "relu",
    a_range: Tuple[float, float] = (-4.0, 4.0),
    figsize: Tuple[float, float] = (7.0, 3.8),
) -> None:
    """Plot activation function h(a) and derivative d/da across a range."""
    a = torch.linspace(a_range[0], a_range[1], 300, requires_grad=True)
    act_fns = {
        "relu": torch.relu,
        "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1),
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }
    fn = act_fns.get(act_name.lower(), torch.relu)
    y = fn(a)
    y.sum().backward()
    grad = a.grad.detach().numpy()

    plt.figure(figsize=figsize)
    plt.plot(a.detach().numpy(), y.detach().numpy(), lw=2.5, color="royalblue", label=f"{act_name}(a)")
    plt.plot(a.detach().numpy(), grad, lw=2.0, linestyle="--", color="firebrick", label="d/da")
    plt.axhline(0, color="gray", lw=0.8, alpha=0.6)
    plt.axvline(0, color="gray", lw=0.8, alpha=0.6)
    plt.title(f"Activation Function: {act_name.upper()} & Derivative", fontsize=12)
    plt.xlabel("Activation input (a)")
    plt.ylabel("Output h(a)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sobel_filtered(
    h_filtered: Any,
    v_filtered: Any,
    filter_type: str = "both",
    figsize: Tuple[float, float] = (7.0, 3.5),
) -> None:
    """Render original and Sobel edge-filtered images."""
    if filter_type == "horizontal":
        plt.figure(figsize=(3.5, 3.5))
        plt.imshow(h_filtered, cmap="gray")
        plt.title("Horizontal Edges (Sobel)", fontsize=11)
        plt.axis("off")
    elif filter_type == "vertical":
        plt.figure(figsize=(3.5, 3.5))
        plt.imshow(v_filtered, cmap="gray")
        plt.title("Vertical Edges (Sobel)", fontsize=11)
        plt.axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(h_filtered, cmap="gray")
        axes[0].set_title("Horizontal Edges", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(v_filtered, cmap="gray")
        axes[1].set_title("Vertical Edges", fontsize=11)
        axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def annotate(first_arg: Any, second_arg: Any, **kwargs: Any) -> None:
    """
    Polymorphic annotation helper (backward-compatible):
    1. CNN Mode: When called with (im, ax), annotates numeric pixel values onto image patches.
    2. Regression Mode: When called with (x, y, **kws), annotates Pearson correlation onto pairgrid.
    """
    if hasattr(second_arg, "text") and hasattr(first_arg, "shape") and len(first_arg.shape) >= 2:
        im = first_arg
        ax = second_arg
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                val = im[i, j].item() if hasattr(im[i, j], "item") else float(im[i, j])
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="r",
                    fontsize=14,
                    weight="bold",
                )
    else:
        x = first_arg
        y = second_arg
        r, p = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate(f"r = {r:.2f}, p = {p:.3f} ", xy=(0.1, 1), xycoords=ax.transAxes)


def plot_correlation_matrix_and_pairgrid(data: Any) -> None:
    """
    Render empirical pairwise relationships and correlation heatmap for dataframe.
    Cross-module compatibility helper.
    """
    p = sns.PairGrid(data, diag_sharey=False)
    p.map_upper(annotate)
    p.map_diag(sns.histplot)
    p.map_diag(sns.kdeplot)
    p.map_lower(sns.scatterplot)
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Empirical Correlation Matrix")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Sequences, Attention & Relational Graphs (11_rnn, 12_transformers, 13_gnn)
# ---------------------------------------------------------------------------

def plot_time_series(
    time: Sequence[Any],
    series: Sequence[Any],
    format: str = "-",
    start: int = 0,
    end: Optional[int] = None,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
) -> None:
    """Plot time-series sequences with optional labeling."""
    plt.figure(figsize=figsize)
    plt.plot(time[start:end], series[start:end], format, label=label)
    if label:
        plt.legend(loc="upper left")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_series(
    time: Sequence[Any],
    series: Sequence[Any],
    format: str = "-",
    start: int = 0,
    end: Optional[int] = None,
    label: Optional[str] = None,
) -> None:
    """Plot time-series sequences (backward-compatible alias)."""
    plot_time_series(time, series, format=format, start=start, end=end, label=label)


def plot_attention_heatmap(
    weights: Any,
    tokens: Optional[Sequence[str]] = None,
    apply_causal_mask: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (5.5, 4.2),
) -> None:
    """Render self-attention weights heatmap."""
    w_np = weights.detach().cpu().numpy() if hasattr(weights, "detach") else np.asarray(weights)
    if w_np.ndim > 2:
        w_np = w_np.squeeze()

    plt.figure(figsize=figsize)
    toks = list(tokens) if tokens is not None else True
    sns.heatmap(
        w_np,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=toks,
        yticklabels=toks,
        cbar=False,
    )
    if title:
        plt.title(title, fontsize=11)
    else:
        plt.title(f"Self-Attention Weights (Causal Mask: {apply_causal_mask})", fontsize=11)
    plt.xlabel("Key Tokens (Content)")
    plt.ylabel("Query Tokens (Focus)")
    plt.tight_layout()
    plt.show()


def plot_graph(G: Any, classes: Optional[Sequence[Any]] = None) -> None:
    """Render PyTorch Geometric graph using NetworkX circular layout."""
    if PYG_AVAILABLE:
        G_nx = to_networkx(G, to_undirected=True)
    else:
        G_nx = G
    nx.draw_networkx(
        G_nx,
        pos=nx.circular_layout(G_nx),
        with_labels=False,
        node_color=classes,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_embedding(
    t: Any,
    classes: Any,
    epoch: Optional[int] = None,
    loss: Optional[Any] = None,
) -> None:
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
    plt.tight_layout()
    plt.show()
