"""
Unified class-based and functional helper utilities for Deep Neural Architectures (ML-DL / 02_deep_learning).
Provides structured, decoupled visualization routines for:
1. Regression, Curve Fitting & Basis Expansions (04_regression)
2. Classification & Decision Boundaries (05_classification, 06_deep_networks)
3. Optimization Landscapes & Convergence Dynamics (07_gradient_descent, 08_backpropagation)
4. Deep Architecture Diagnostics, Convolutions & Sequences (09_regularization, 10_convolutional_networks, 11_sequence_models, 12_transformers, 13_graph_neural_networks)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch

try:
    from torch_geometric.utils import to_networkx
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Core Shared Foundation (Asset Resolution, Aesthetics, Base Plotter)
# ---------------------------------------------------------------------------

from .shared import (  # noqa: F401 -- re-exported for notebooks as hp.<name>
    COURSE_RAW_URL,
    DATA_BASE_URL,
    BasePlotter,
    annotate,
    get_data_path,
    get_file,
    get_url,
    plot_residuals,
    print_math,
    setup_theme,
)

_to_numpy = BasePlotter.to_numpy  # short internal name used by the plotters below



# ---------------------------------------------------------------------------
# 2. Regression, Curve Fitting & Basis Expansions (04_regression)
# ---------------------------------------------------------------------------

class DeepRegressionPlotter(BasePlotter):
    """Encapsulated plotting routines for single-layer linear regression and regularized models."""

    @classmethod
    def plot_synthetic_regression_data(
        cls,
        x_dense: torch.Tensor,
        t_dense: torch.Tensor,
        x_train: torch.Tensor,
        t_train: torch.Tensor,
        title: str = "Synthetic Regression Dataset: True Signal and Noisy Observations",
        figsize: Tuple[float, float] = (7.5, 3.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Plot ground-truth signal and training observations."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        x_d = _to_numpy(x_dense).ravel()
        t_d = _to_numpy(t_dense).ravel()
        x_tr = _to_numpy(x_train).ravel()
        t_tr = _to_numpy(t_train).ravel()

        ax_plot.plot(x_d, t_d, "forestgreen", lw=2, label=r"Ground Truth: $\sin(2\pi x)$")
        ax_plot.scatter(
            x_tr,
            t_tr,
            color="royalblue",
            s=50,
            alpha=0.85,
            label=f"Training Observations ($N = {len(x_tr)}$)",
        )
        ax_plot.set_title(title, fontsize=11.5)
        ax_plot.set_xlabel("Input Feature ($x$)", fontsize=11)
        ax_plot.set_ylabel("Target ($t$)", fontsize=11)
        ax_plot.set_ylim(-1.5, 1.5)
        ax_plot.legend(loc="upper right", fontsize=9.5)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_regression_fit(
        cls,
        x_train: torch.Tensor,
        t_train: torch.Tensor,
        x_eval: torch.Tensor,
        y_eval: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        residuals: bool = False,
        title: Optional[str] = None,
        xlabel: str = "Input Feature ($x$)",
        ylabel: str = "Target ($t$)",
        ylim: Optional[Tuple[float, float]] = (-1.5, 1.5),
        figsize: Tuple[float, float] = (7.5, 3.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Decoupled 1D regression curve visualizer with observations, fitted curve, and optional residual drop-lines."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        x_tr = _to_numpy(x_train).ravel()
        t_tr = _to_numpy(t_train).ravel()
        x_ev = _to_numpy(x_eval).ravel()
        y_ev = _to_numpy(y_eval).ravel()

        if y_true is not None:
            y_tr_true = _to_numpy(y_true).ravel()
            ax_plot.plot(x_ev, y_tr_true, color="forestgreen", lw=2, label="Ground Truth")

        ax_plot.scatter(
            x_tr,
            t_tr,
            color="royalblue",
            s=45,
            alpha=0.85,
            label=f"Observations ($N={len(x_tr)}$)",
        )
        ax_plot.plot(x_ev, y_ev, color="firebrick", lw=2.2, label="Model Prediction")

        if residuals:
            y_interp = np.interp(x_tr, x_ev, y_ev)
            ax_plot.vlines(
                x_tr,
                t_tr,
                y_interp,
                colors="royalblue",
                linestyles="--",
                alpha=0.5,
                label="Residuals",
            )

        if title:
            ax_plot.set_title(title, fontsize=11.5)
        ax_plot.set_xlabel(xlabel, fontsize=11)
        ax_plot.set_ylabel(ylabel, fontsize=11)
        if ylim is not None:
            ax_plot.set_ylim(ylim)
        ax_plot.legend(loc="best", fontsize=9.5)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_polynomial_fits(
        cls,
        x_dense: torch.Tensor,
        t_dense: torch.Tensor,
        x_train: torch.Tensor,
        t_train: torch.Tensor,
        fitted_models: Dict[int, torch.Tensor],
        degrees: Sequence[int],
        figsize: Tuple[float, float] = (9.5, 5.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot 2x2 grid of polynomial curve fits for varying degrees M."""
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        x_d = _to_numpy(x_dense).ravel()
        t_d = _to_numpy(t_dense).ravel()
        x_tr = _to_numpy(x_train).ravel()
        t_tr = _to_numpy(t_train).ravel()

        for ax, M in zip(axes.ravel(), degrees):
            w_fit = fitted_models[M]
            y_dense = torch.matmul(torch.vander(x_dense, N=M + 1, increasing=True), w_fit)
            y_d = _to_numpy(y_dense).ravel()

            ax.plot(x_d, t_d, "forestgreen", lw=1.5, label="True")
            ax.scatter(x_tr, t_tr, color="royalblue", s=35, alpha=0.85)
            ax.plot(x_d, y_d, "firebrick", lw=2, label=f"$M={M}$")
            ax.set_title(f"Polynomial Degree $M={M}$", fontsize=10.5)
            ax.set_ylim(-1.6, 1.6)
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(True)

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_polynomial_extrapolation(
        cls,
        x_train: torch.Tensor,
        t_train: torch.Tensor,
        w: torch.Tensor,
        degree: Optional[int] = None,
        x_range: Tuple[float, float] = (0.0, 1.0),
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.5, 3.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Optional[plt.Axes]:
        """Decoupled polynomial curve visualizer with interval extrapolation support."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)

        if "x_min" in kwargs and "x_max" in kwargs:
            x_min, x_max = float(kwargs["x_min"]), float(kwargs["x_max"])
        elif isinstance(x_range, (tuple, list)) and len(x_range) == 2:
            x_min, x_max = float(x_range[0]), float(x_range[1])
        else:
            x_min, x_max = 0.0, 1.0

        w_t = w if isinstance(w, torch.Tensor) else torch.as_tensor(w, dtype=torch.float32)
        deg = degree if degree is not None else (len(w_t) - 1)

        x_grid = torch.linspace(x_min, x_max, 400)
        Phi_grid = torch.vander(x_grid, N=deg + 1, increasing=True)
        y_grid = torch.matmul(Phi_grid, w_t)

        x_grid_np = _to_numpy(x_grid).ravel()
        y_grid_np = _to_numpy(y_grid).ravel()
        x_tr_np = _to_numpy(x_train).ravel()
        t_tr_np = _to_numpy(t_train).ravel()
        y_true_np = np.sin(2.0 * np.pi * x_grid_np)

        ax_plot.plot(x_grid_np, y_true_np, color="forestgreen", lw=1.8, label=r"Ground Truth: $\sin(2\pi x)$")
        ax_plot.scatter(
            x_tr_np,
            t_tr_np,
            color="royalblue",
            s=45,
            alpha=0.85,
            label=f"Observations ($N={len(x_tr_np)}$)",
            zorder=4,
        )
        ax_plot.plot(x_grid_np, y_grid_np, color="firebrick", lw=2.2, label=f"Polynomial ($M={deg}$)")

        if x_min < 0.0 or x_max > 1.0:
            ax_plot.axvspan(0.0, 1.0, color="royalblue", alpha=0.08, label="Training Domain [0, 1]")
            ax_plot.set_ylim(-3.5, 3.5)
        else:
            ax_plot.set_ylim(-1.6, 1.6)

        ax_plot.set_xlim(x_min, x_max)
        plot_title = title if title is not None else f"Polynomial Fit ($M={deg}$) on Interval $[{x_min:.1f}, {x_max:.1f}]$"
        ax_plot.set_title(plot_title, fontsize=11.5)
        ax_plot.set_xlabel("Input Feature ($x$)", fontsize=11)
        ax_plot.set_ylabel("Target ($t$)", fontsize=11)
        ax_plot.legend(loc="upper right", fontsize=8.5)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_dataset_size_remedy(
        cls,
        x_dense: torch.Tensor,
        t_dense: torch.Tensor,
        models_by_N: Sequence[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
        figsize: Tuple[float, float] = (9.5, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot 1x2 panel illustrating mitigating power of sample size N for flexible model M=9."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        x_d = _to_numpy(x_dense).ravel()
        t_d = _to_numpy(t_dense).ravel()

        for ax, (N_val, x_sc, t_sc, y_dense_sc) in zip(axes, models_by_N):
            x_s = _to_numpy(x_sc).ravel()
            t_s = _to_numpy(t_sc).ravel()
            y_s = _to_numpy(y_dense_sc).ravel()

            ax.plot(x_d, t_d, "forestgreen", lw=1.5, label="True")
            ax.scatter(x_s, t_s, color="royalblue", s=25, alpha=0.75)
            ax.plot(x_d, y_s, "firebrick", lw=2, label="$M=9$")
            ax.set_title(f"Model $M = 9$ with $N = {N_val}$ Samples", fontsize=10.5)
            ax.set_ylim(-1.6, 1.6)
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(True)

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_regularized_fits(
        cls,
        x_dense: torch.Tensor,
        t_dense: torch.Tensor,
        x_train: torch.Tensor,
        t_train: torch.Tensor,
        reg_fits: Dict[Any, torch.Tensor],
        reg_lambdas: Sequence[Any],
        figsize: Tuple[float, float] = (9.5, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot 1x2 panel of regularized polynomial fits for M=9."""
        Phi_dense_10 = torch.vander(x_dense, N=10, increasing=True)
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        titles = [r"$\ln\lambda = -18$ (Optimal Regularization)", r"$\ln\lambda = 0$ (Heavy Damping)"]
        x_d = _to_numpy(x_dense).ravel()
        t_d = _to_numpy(t_dense).ravel()
        x_tr = _to_numpy(x_train).ravel()
        t_tr = _to_numpy(t_train).ravel()

        for ax, lam, title in zip(axes, reg_lambdas, titles):
            w_reg = reg_fits[lam]
            y_reg_dense = torch.matmul(Phi_dense_10, w_reg)
            y_reg = _to_numpy(y_reg_dense).ravel()

            ax.plot(x_d, t_d, "forestgreen", lw=1.5, label="True")
            ax.scatter(x_tr, t_tr, color="royalblue", s=35, alpha=0.85)
            ax.plot(x_d, y_reg, "firebrick", lw=2, label="Fit")
            ax.set_title(title, fontsize=10.5)
            ax.set_ylim(-1.6, 1.6)
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(True)

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_basis_functions(
        cls,
        x_eval: torch.Tensor,
        centers: Sequence[float],
        s: float = 0.15,
        figsize: Tuple[float, float] = (10.5, 3.4),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot 1x3 panel displaying canonical families of basis functions (polynomial, RBF, sigmoidal)."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        x_ev = _to_numpy(x_eval).ravel()

        for p in range(4):
            axes[0].plot(x_ev, x_ev ** p, lw=2, label=f"$x^{p}$")
        axes[0].set_title("Polynomial Basis", fontsize=10.5)
        axes[0].legend(fontsize=8.5)

        for mu in centers:
            mu_val = cls.to_scalar(mu)
            axes[1].plot(x_ev, np.exp(-0.5 * ((x_ev - mu_val) / s) ** 2), lw=2)
        axes[1].set_title("Gaussian RBF Basis", fontsize=10.5)

        for mu in centers:
            mu_val = cls.to_scalar(mu)
            axes[2].plot(x_ev, 1.0 / (1.0 + np.exp(-(x_ev - mu_val) / s)), lw=2)

        axes[2].set_title("Sigmoidal Basis", fontsize=10.5)

        for ax in axes:
            ax.set_xlabel("$x$")
            ax.set_ylim(-0.1, 1.2)
            ax.grid(True)

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_bias_variance_curve(
        cls,
        ln_lambdas: Sequence[float],
        bias2_list: Sequence[float],
        var_list: Sequence[float],
        total_bv: Sequence[float],
        xlabel: str = r"$\ln\lambda$",
        ylabel: str = "Error",
        ylim: Tuple[float, float] = (0, 0.12),
        figsize: Tuple[float, float] = (7.5, 3.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Plot canonical U-shaped bias-variance trade-off curve vs ln(lambda)."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        ax_plot.plot(ln_lambdas, bias2_list, "royalblue", lw=2.5, label=r"$(\mathrm{bias})^2$")
        ax_plot.plot(ln_lambdas, var_list, "firebrick", lw=2.5, label=r"$\mathrm{variance}$")
        ax_plot.plot(ln_lambdas, total_bv, "purple", lw=2.5, linestyle="--", label=r"$(\mathrm{bias})^2 + \mathrm{variance}$")
        ax_plot.set_title(r"The Bias–Variance Trade-Off vs $\ln\lambda$", fontsize=11.5)
        ax_plot.set_xlabel(xlabel, fontsize=11)
        ax_plot.set_ylabel(ylabel, fontsize=11)
        ax_plot.set_ylim(ylim)
        ax_plot.legend(loc="upper center", fontsize=9.0)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_ensemble_fits(
        cls,
        x_dense_ev: torch.Tensor,
        y_true_ev: torch.Tensor,
        ensembles: Dict[float, Tuple[Sequence[torch.Tensor], torch.Tensor]],
        figsize: Tuple[float, float] = (11, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Render 1x3 panel of model ensembles across High, Optimal, and Low regularization."""
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        panel_specs = [
            (2.0, r"High Regularization ($\ln\lambda = 2.0$)\nHigh Bias, Low Variance"),
            (-0.5, r"Optimal Balance ($\ln\lambda = -0.5$)\nBalanced Bias & Variance"),
            (-4.0, r"Low Regularization ($\ln\lambda = -4.0$)\nLow Bias, High Variance"),
        ]
        x_ev = _to_numpy(x_dense_ev).ravel()
        y_tr = _to_numpy(y_true_ev).ravel()

        for ax, (ln_lam, title) in zip(axes, panel_specs):
            preds_all, mean_pred = ensembles[ln_lam]
            mean_p = _to_numpy(mean_pred).ravel()

            for l in range(min(20, len(preds_all))):
                pred_l = _to_numpy(preds_all[l]).ravel()
                ax.plot(x_ev, pred_l, color="firebrick", alpha=0.3, lw=0.9)

            ax.plot(x_ev, mean_p, color="darkred", lw=2.4, label=r"Ensemble Mean $\bar{y}(x)$")
            ax.plot(x_ev, y_tr, color="forestgreen", lw=2.0, linestyle="--", label="Ground Truth")
            ax.set_title(title, fontsize=10.0)
            ax.set_xlabel("$x$")
            ax.set_ylim(-1.6, 1.6)
            ax.grid(True)
            if ax == axes[0]:
                ax.set_ylabel("$y(x)$")
                ax.legend(loc="lower left", fontsize=8.0)

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_l1_l2_sparsity_geometry(
        cls,
        figsize: Tuple[float, float] = (9.5, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
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

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_l1_l2_outlier_comparison(
        cls,
        x_clean: torch.Tensor,
        t_corrupt: torch.Tensor,
        x_plot: torch.Tensor,
        y_l2_plot: torch.Tensor,
        y_l1_plot: torch.Tensor,
        figsize: Tuple[float, float] = (7.5, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot comparison of L2 squared loss vs L1 absolute loss on corrupted data."""
        fig, ax = cls.create_canvas(figsize=figsize)
        x_c = _to_numpy(x_clean).ravel()
        t_c = _to_numpy(t_corrupt).ravel()
        x_p = _to_numpy(x_plot).ravel()
        y_l2 = _to_numpy(y_l2_plot).ravel()
        y_l1 = _to_numpy(y_l1_plot).ravel()

        ax.scatter(x_c, t_c, color="royalblue", s=40, label="Data with Sensor Spikes")
        ax.plot(x_p, y_l2, "firebrick", lw=2, label=r"$L_2$ Squared Loss (Skewed)")
        ax.plot(
            x_p,
            y_l1,
            "forestgreen",
            lw=2.5,
            linestyle="--",
            label=r"$L_1$ Absolute Loss (Robust Median)",
        )
        ax.set_title(r"Decision Theory: Robustness of $L_1$ vs $L_2$ Loss", fontsize=11.5)
        ax.set_xlabel("Input Feature ($x$)")
        ax.set_ylabel("Target ($t$)")
        ax.legend(loc="upper left", fontsize=9.0)
        ax.grid(True)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_streaming_adaline_convergence(
        cls,
        X_synth: torch.Tensor,
        t_synth: torch.Tensor,
        x_line: torch.Tensor,
        Phi_line: torch.Tensor,
        epoch_snapshots: Sequence[Tuple[int, torch.Tensor]],
        w_analytic: torch.Tensor,
        figsize: Tuple[float, float] = (8, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot streaming Adaline regression lines across epochs converging to analytical batch solution."""
        fig, ax = cls.create_canvas(figsize=figsize)
        X_s = _to_numpy(X_synth).ravel()
        t_s = _to_numpy(t_synth).ravel()
        x_l = _to_numpy(x_line).ravel()

        ax.scatter(
            X_s,
            t_s,
            color="royalblue",
            s=45,
            alpha=0.85,
            label=f"Streaming Samples ($N={len(X_s)}$)",
        )

        colors = {"1": "orange", "3": "darkviolet", "15": "crimson"}
        for ep, w_snap in epoch_snapshots[1:]:
            y_snap = Phi_line @ w_snap
            y_sn = _to_numpy(y_snap).ravel()
            ax.plot(
                x_l,
                y_sn,
                color=colors.get(str(ep), "gray"),
                lw=1.8,
                linestyle="--",
                label=f"Adaline Epoch {ep}",
            )

        y_exact = Phi_line @ w_analytic
        y_ex = _to_numpy(y_exact).ravel()
        ax.plot(x_l, y_ex, "black", lw=2.5, label=r"Batch OLS $\mathbf{w}_\mathrm{ML}$")

        ax.set_title("Adaline Online LMS: Progressive Convergence to Batch Least Squares", fontsize=11.5)
        ax.set_xlabel("Input Feature ($x$)")
        ax.set_ylabel("Target ($t$)")
        ax.legend(loc="lower right", fontsize=9.0)
        ax.grid(True)
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 3. Classification & Decision Boundaries (05_classification, 06_deep_networks)
# ---------------------------------------------------------------------------

class DeepClassificationPlotter(BasePlotter):
    """Encapsulated decision boundary visualizers for single-layer and deep classifiers."""

    @classmethod
    def plot_decision_boundary_2d(
        cls,
        predictor: Callable[[np.ndarray], np.ndarray],
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Plot 2D decision boundary meshgrid and data points for classification."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        X_np = cls.to_numpy(X)
        y_np = cls.to_numpy(y, ravel=True)

        xm, ym, grid_pts = cls.create_2d_grid(X_np, padding=0.5, resolution=200)
        p = cls.eval_predictor(predictor, grid_pts, as_tensor_like=X).reshape(xm.shape)

        ax_plot.contourf(xm, ym, p, alpha=0.3, cmap="coolwarm")
        ax_plot.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="coolwarm", edgecolors="k", s=80, lw=1.5)

        if title:
            ax_plot.set_title(title, fontsize=12)
        ax_plot.set_xlabel("$x_1$", fontsize=11.5)
        ax_plot.set_ylabel("$x_2$", fontsize=11.5)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)


# ---------------------------------------------------------------------------
# 4. Optimization Landscapes & Convergence Dynamics (07_gradient_descent, 08_backpropagation)
# ---------------------------------------------------------------------------

class DeepOptimizationPlotter(BasePlotter):
    """Encapsulated loss landscape and convergence trajectory visualizers."""

    @classmethod
    def plot_loss_history(
        cls,
        losses: Sequence[float],
        log_scale: bool = True,
        title: str = "Optimization Convergence",
        xlabel: str = "Iteration",
        ylabel: str = "Loss / Metric",
        figsize: Tuple[float, float] = (8, 4.5),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Plot optimization convergence loss with optional logarithmic scaling."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        ax_plot.plot(losses, lw=2.2, color="royalblue")
        if log_scale:
            ax_plot.set_yscale("log")
        ax_plot.set_title(title, fontsize=12)
        ax_plot.set_xlabel(xlabel, fontsize=11)
        ax_plot.set_ylabel(ylabel, fontsize=11)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_out(cls, out: Sequence[Any]) -> None:
        """Plot optimization convergence loss on a logarithmic scale with formatted summary."""
        grad_val = cls.to_scalar(out[-1])
        print_math(rf"Finished optimization in {len(out)} iterations with final metric $\|\nabla E(\mathbf{{w}})\| = {grad_val:.4e}$")
        cls.plot_loss_history(out, log_scale=True)

    @classmethod
    def plot_idea(
        cls,
        f: Callable[[Any], Any],
        df: Callable[[Any], Any],
        x: float = 1.0,
        h: float = 0.05,
        figsize: Tuple[float, float] = (8, 5),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Illustrate 1D gradient step concept with direction arrow."""
        fig, ax = cls.create_canvas(figsize=figsize)
        sp = np.linspace(-0.5, 1, 100)
        ax.plot(sp, f(sp), lw=2, color="royalblue")
        x_n = x - h * df(x)
        ax.arrow(
            x,
            f(x),
            x_n - x,
            f(x_n) - f(x),
            head_width=0.04,
            length_includes_head=True,
            color="crimson",
            linewidth=3,
        )
        ax.plot(x, f(x), "ro", markersize=8)
        ax.set_title(f"Gradient Step: x={x:.2f} -> x_next={x_n:.2f}", fontsize=12)
        ax.grid(True)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_trace(
        cls,
        f: Callable[[Any], Any],
        trace: Sequence[Any],
        figsize: Tuple[float, float] = (8, 5),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot optimization parameter trajectory over 1D objective surface."""
        fig, ax = cls.create_canvas(figsize=figsize)
        max_bound = max(abs(min(trace)), abs(max(trace))) + 0.5
        domain = torch.arange(-max_bound, max_bound, 0.01)
        f_domain = cls.to_numpy(f(domain))
        ax.plot(domain.numpy(), f_domain, "b-", lw=2, label="Objective f(x)")
        trace_vals = [cls.to_scalar(f(x)) for x in trace]
        ax.plot(trace, trace_vals, "-ro", markersize=6, label="Trajectory")
        ax.legend()
        ax.grid(True)
        return cls.finalize(fig, show=show)


    @classmethod
    def plot_grad(
        cls,
        f: Callable[[Any, Any], Any],
        x: float,
        y: float,
        x1: float,
        y1: float,
        angle: float = 60.0,
        figsize: Tuple[float, float] = (8.0, 8.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot 3D loss surface and single gradient step projection."""
        fig = plt.figure(figsize=figsize)
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

        print_math(rf"Initial state $\mathbf{{w}}_0 = [{x:.2f}, {y:.2f}]^\mathrm{{T}}, E(\mathbf{{w}}_0) = {f(x, y):.2f}$; Step $\mathbf{{w}}_1 = [{x1:.2f}, {y1:.2f}]^\mathrm{{T}}, E(\mathbf{{w}}_1) = {f(x1, y1):.2f}$")
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 5. Deep Diagnostics, Convolutions, Sequences & Graphs
# ---------------------------------------------------------------------------

class DeepArchitecturePlotter(BasePlotter):
    """Encapsulated diagnostics for activations, feature maps, attention, sequences, and graphs."""

    @classmethod
    def plot_activation_curves(
        cls,
        act_name: str = "relu",
        a_range: Tuple[float, float] = (-4.0, 4.0),
        figsize: Tuple[float, float] = (7.0, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
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
        grad = cls.to_numpy(a.grad)
        a_np = cls.to_numpy(a)
        y_np = cls.to_numpy(y)

        fig, ax = cls.create_canvas(figsize=figsize)
        ax.plot(a_np, y_np, lw=2.5, color="royalblue", label=f"{act_name}(a)")
        ax.plot(a_np, grad, lw=2.0, linestyle="--", color="firebrick", label="d/da")
        ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.6)
        ax.set_title(f"Activation Function: {act_name.upper()} & Derivative", fontsize=12)
        ax.set_xlabel("Activation input (a)")
        ax.set_ylabel("Output h(a)")
        ax.legend(loc="upper left")
        ax.grid(True)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_sobel_filtered(
        cls,
        h_filtered: torch.Tensor | np.ndarray,
        v_filtered: torch.Tensor | np.ndarray,
        filter_type: str = "both",
        figsize: Tuple[float, float] = (7.0, 3.5),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Render original and Sobel edge-filtered images."""
        h_np = cls.to_numpy(h_filtered)
        v_np = cls.to_numpy(v_filtered)

        if filter_type == "horizontal":
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(h_np, cmap="gray")
            ax.set_title("Horizontal Edges (Sobel)", fontsize=11)
            ax.axis("off")
        elif filter_type == "vertical":
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(v_np, cmap="gray")
            ax.set_title("Vertical Edges (Sobel)", fontsize=11)
            ax.axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            axes[0].imshow(h_np, cmap="gray")
            axes[0].set_title("Horizontal Edges", fontsize=11)
            axes[0].axis("off")
            axes[1].imshow(v_np, cmap="gray")
            axes[1].set_title("Vertical Edges", fontsize=11)
            axes[1].axis("off")

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_time_series(
        cls,
        time: Sequence[Any],
        series: Sequence[Any],
        format: str = "-",
        start: int = 0,
        end: Optional[int] = None,
        label: Optional[str] = None,
        figsize: Tuple[float, float] = (15, 5),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot time-series sequences with optional labeling."""
        fig, ax = cls.create_canvas(figsize=figsize)
        ax.plot(time[start:end], series[start:end], format, label=label)
        if label:
            ax.legend(loc="upper left")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_attention_heatmap(
        cls,
        weights: torch.Tensor | np.ndarray,
        tokens: Optional[Sequence[str]] = None,
        apply_causal_mask: bool = False,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5.5, 4.2),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Render self-attention weights heatmap."""
        w_np = cls.to_numpy(weights)
        if w_np.ndim > 2:
            w_np = w_np.squeeze()

        fig, ax = cls.create_canvas(figsize=figsize)
        toks = list(tokens) if tokens is not None else True
        sns.heatmap(
            w_np,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=toks,
            yticklabels=toks,
            cbar=False,
            ax=ax,
        )
        if title:
            ax.set_title(title, fontsize=11)
        else:
            ax.set_title(f"Self-Attention Weights (Causal Mask: {apply_causal_mask})", fontsize=11)
        ax.set_xlabel("Key Tokens (Content)")
        ax.set_ylabel("Query Tokens (Focus)")
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_graph(
        cls,
        G: Any,
        classes: Optional[Sequence[Any]] = None,
        figsize: Tuple[float, float] = (7.0, 7.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Render PyTorch Geometric / NetworkX graph using circular layout."""
        if PYG_AVAILABLE and not isinstance(G, nx.Graph):
            G_nx = to_networkx(G, to_undirected=True)
        else:
            G_nx = G

        fig, ax = cls.create_canvas(figsize=figsize)
        nx.draw_networkx(
            G_nx,
            pos=nx.circular_layout(G_nx),
            with_labels=False,
            node_color=classes,
            ax=ax,
        )
        ax.axis("off")
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_embedding(
        cls,
        t: torch.Tensor | np.ndarray,
        classes: Any,
        epoch: Optional[int] = None,
        loss: Optional[Any] = None,
        figsize: Tuple[float, float] = (7.0, 5.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Render 2D graph node embeddings colored by class labels."""
        t_arr = cls.to_numpy(t)
        fig, ax = cls.create_canvas(figsize=figsize)
        ax.scatter(t_arr[:, 0], t_arr[:, 1], s=140, c=classes, cmap="tab10", edgecolors="k", alpha=0.85)
        if epoch is not None and loss is not None:
            loss_val = cls.to_scalar(loss)
            ax.set_xlabel(f"Epoch: {epoch}, Loss: {loss_val:.4f}", fontsize=14)
        ax.grid(True)
        return cls.finalize(fig, show=show)



# ---------------------------------------------------------------------------
# 6. Backward-Compatible Functional Facades
# ---------------------------------------------------------------------------

# Regression Facades
plot_synthetic_regression_data = DeepRegressionPlotter.plot_synthetic_regression_data
plot_regression_fit = DeepRegressionPlotter.plot_regression_fit
plot_polynomial_fits = DeepRegressionPlotter.plot_polynomial_fits
plot_polynomial_extrapolation = DeepRegressionPlotter.plot_polynomial_extrapolation
plot_dataset_size_remedy = DeepRegressionPlotter.plot_dataset_size_remedy
plot_regularized_fits = DeepRegressionPlotter.plot_regularized_fits
plot_basis_functions = DeepRegressionPlotter.plot_basis_functions
plot_bias_variance_curve = DeepRegressionPlotter.plot_bias_variance_curve
plot_ensemble_fits = DeepRegressionPlotter.plot_ensemble_fits
plot_l1_l2_sparsity_geometry = DeepRegressionPlotter.plot_l1_l2_sparsity_geometry
plot_l1_l2_outlier_comparison = DeepRegressionPlotter.plot_l1_l2_outlier_comparison
plot_streaming_adaline_convergence = DeepRegressionPlotter.plot_streaming_adaline_convergence

# Classification Facades
plot_decision_boundary_2d = DeepClassificationPlotter.plot_decision_boundary_2d
plot_binary = DeepClassificationPlotter.plot_decision_boundary_2d

# Optimization Facades
plot_loss_history = DeepOptimizationPlotter.plot_loss_history
plot_out = DeepOptimizationPlotter.plot_out
g_idea = DeepOptimizationPlotter.plot_idea
plot_trace = DeepOptimizationPlotter.plot_trace
plot_grad = DeepOptimizationPlotter.plot_grad

# Architecture & Diagnostics Facades
plot_activation_curves = DeepArchitecturePlotter.plot_activation_curves
plot_sobel_filtered = DeepArchitecturePlotter.plot_sobel_filtered
plot_time_series = DeepArchitecturePlotter.plot_time_series
plot_series = DeepArchitecturePlotter.plot_time_series
plot_attention_heatmap = DeepArchitecturePlotter.plot_attention_heatmap
plot_graph = DeepArchitecturePlotter.plot_graph
plot_embedding = DeepArchitecturePlotter.plot_embedding

# Conversion & Helper Facades
to_numpy = BasePlotter.to_numpy
to_scalar = BasePlotter.to_scalar
ensure_2d = BasePlotter.ensure_2d
eval_predictor = BasePlotter.eval_predictor
get_boundaries = BasePlotter.get_boundaries
create_2d_grid = BasePlotter.create_2d_grid

