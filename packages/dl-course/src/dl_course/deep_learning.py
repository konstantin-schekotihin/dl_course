"""
Unified class-based and functional helper utilities for Deep Neural Architectures (ML-DL / 02_deep_learning).
Provides structured, decoupled visualization routines for:
1. Regression, Curve Fitting & Basis Expansions (04_regression)
2. Classification & Decision Boundaries (05_classification, 06_deep_networks)
3. Optimization Landscapes & Convergence Dynamics (07_gradient_descent, 08_backpropagation)
4. Deep Architecture Diagnostics, Convolutions & Sequences (09_regularization, 10_convolutional_networks, 11_sequence_models, 12_transformers, 13_graph_neural_networks)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

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
    format_number,
    format_vector,
    plot_confusion_matrix,
    plot_curves,
    plot_decision_surface,
    plot_fitted_curves,
    plot_function_panels,
    plot_image_grid,
    plot_prediction,
    plot_residuals,
    plot_roc_curves,
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
        x_train: Any,
        t_train: Any,
        x_eval: Any,
        y_eval: Any,
        y_true: Optional[Any] = None,
        residuals: bool = False,
        band: Optional[Tuple[Any, Any]] = None,
        band_label: str = r"$y \pm 2\sigma_\mathrm{ML}$",
        title: Optional[str] = None,
        xlabel: str = "$x$",
        ylabel: str = "$t$",
        ylim: Optional[Tuple[float, float]] = (-1.5, 1.5),
        figsize: Tuple[float, float] = (7.5, 3.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Observations, a fitted curve pre-computed on x_eval, optional h, residual drop-lines and a (lower, upper) band."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        x_tr, t_tr = _to_numpy(x_train).ravel(), _to_numpy(t_train).ravel()
        x_ev, y_ev = _to_numpy(x_eval).ravel(), _to_numpy(y_eval).ravel()
        if band is not None:
            ax_plot.fill_between(x_ev, _to_numpy(band[0]).ravel(), _to_numpy(band[1]).ravel(), color="firebrick",
                                 alpha=0.15, label=band_label)
        if y_true is not None:
            ax_plot.plot(x_ev, _to_numpy(y_true).ravel(), color="forestgreen", lw=2, label=r"$h(x)$")
        ax_plot.scatter(x_tr, t_tr, color="royalblue", s=45, alpha=0.85, label=f"Observations ($N = {len(x_tr)}$)")
        ax_plot.plot(x_ev, y_ev, color="firebrick", lw=2.2, label="Model $y(x, \\mathbf{w})$")
        if residuals:
            y_at_train = np.interp(x_tr, x_ev, y_ev)  # drawing only: the curve's height at the observations
            ax_plot.vlines(x_tr, t_tr, y_at_train, colors="royalblue", linestyles="--", alpha=0.6, label="Residuals")
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
    def plot_conditional_density(
        cls,
        x_grid: Any,
        y_grid: Any,
        x0: float,
        t_grid: Any,
        density: Any,
        y0: float,
        width: float = 0.25,
        figsize: Tuple[float, float] = (7.5, 3.8),
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Regression function f*(x) = E[t | x] on the grid and the conditional density p(t | x0) drawn sideways at x0, its mean y0 marked (after Bishop & Bishop 2024, Fig. 4.5); all values pre-computed."""
        fig, ax_plot = cls.create_canvas(figsize=figsize)
        ax_plot.plot(_to_numpy(x_grid).ravel(), _to_numpy(y_grid).ravel(), color="firebrick", lw=2.2,
                     label=r"$f^\star(x) = \mathbb{E}[t \mid x]$")
        dens = _to_numpy(density).ravel()
        t_g = _to_numpy(t_grid).ravel()
        ax_plot.plot(x0 + width * dens / dens.max(), t_g, color="royalblue", lw=2.0, label=r"$p(t \mid x_0)$")
        ax_plot.axvline(x0, color="gray", lw=1.0, linestyle=":")
        ax_plot.plot([x0], [y0], "o", color="firebrick", ms=7)
        ax_plot.annotate(r"$f^\star(x_0)$", (x0, y0), xytext=(8, -14), textcoords="offset points", fontsize=11)
        ax_plot.set_xlabel("$x$", fontsize=11)
        ax_plot.set_ylabel("$t$", fontsize=11)
        ax_plot.legend(loc="upper right", fontsize=9.5)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_polynomial_extrapolation(
        cls,
        x_train: Any,
        t_train: Any,
        x_grid: Any,
        y_grid: Any,
        h_grid: Optional[Any] = None,
        degree: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.5, 3.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Observations, the generating function h on the grid and a polynomial fit pre-computed on the grid; shades the training interval [0, 1] when the grid leaves it."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        x_g, y_g = _to_numpy(x_grid).ravel(), _to_numpy(y_grid).ravel()
        x_min, x_max = float(x_g.min()), float(x_g.max())
        if h_grid is not None:
            ax_plot.plot(x_g, _to_numpy(h_grid).ravel(), color="forestgreen", lw=1.8, label=r"$h(x) = \sin(2\pi x)$")
        x_tr = _to_numpy(x_train).ravel()
        ax_plot.scatter(x_tr, _to_numpy(t_train).ravel(), color="royalblue", s=45, alpha=0.85,
                        label=f"Observations ($N = {len(x_tr)}$)", zorder=4)
        label = "Polynomial" if degree is None else f"Polynomial ($M = {degree}$)"
        ax_plot.plot(x_g, y_g, color="firebrick", lw=2.2, label=label)
        if x_min < 0.0 or x_max > 1.0:
            ax_plot.axvspan(0.0, 1.0, color="royalblue", alpha=0.08, label="Training interval $[0, 1]$")
            ax_plot.set_ylim(-3.5, 3.5)
        else:
            ax_plot.set_ylim(-1.6, 1.6)
        ax_plot.set_xlim(x_min, x_max)
        ax_plot.set_title(title or label, fontsize=11.5)
        ax_plot.set_xlabel("$x$", fontsize=11)
        ax_plot.set_ylabel("$t$", fontsize=11)
        ax_plot.legend(loc="upper right", fontsize=8.5)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_dataset_size_remedy(
        cls,
        x_dense: Any,
        h_dense: Any,
        models_by_N: Sequence[Tuple[int, Any, Any, Any]],
        M: int,
        figsize: Tuple[float, float] = (9.5, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """One panel per data-set size: observations, h and the fit pre-computed on x_dense, given as (N, x, t, y_dense)."""
        fig, axes = plt.subplots(1, len(models_by_N), figsize=figsize, sharey=True, squeeze=False)
        x_d, h_d = _to_numpy(x_dense).ravel(), _to_numpy(h_dense).ravel()
        for ax, (N_val, x_sc, t_sc, y_dense_sc) in zip(axes[0], models_by_N):
            ax.plot(x_d, h_d, "forestgreen", lw=1.5, label=r"$h(x)$")
            ax.scatter(_to_numpy(x_sc).ravel(), _to_numpy(t_sc).ravel(), color="royalblue", s=25, alpha=0.75)
            ax.plot(x_d, _to_numpy(y_dense_sc).ravel(), "firebrick", lw=2, label=f"$M = {M}$")
            ax.set_title(f"$M = {M}$, $N = {N_val}$", fontsize=10.5)
            ax.set_xlabel("$x$")
            ax.set_ylim(-1.6, 1.6)
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(True)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_bias_variance_curve(
        cls,
        lambdas: Sequence[float],
        bias2: Sequence[float],
        variance: Sequence[float],
        total: Sequence[float],
        test_error: Optional[Sequence[float]] = None,
        xlabel: str = r"$\lambda$ (logarithmic axis)",
        ylabel: str = "Error",
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (7.5, 3.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Squared bias, variance, their sum and optionally the average test error, all pre-computed per lambda (logarithmic axis)."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        ax_plot.plot(lambdas, bias2, "royalblue", lw=2.5, label=r"$(\mathrm{bias})^2$")
        ax_plot.plot(lambdas, variance, "firebrick", lw=2.5, label=r"$\mathrm{variance}$")
        ax_plot.plot(lambdas, total, "purple", lw=2.5, linestyle="--", label=r"$(\mathrm{bias})^2 + \mathrm{variance}$")
        if test_error is not None:
            ax_plot.plot(lambdas, test_error, "black", lw=2.0, label="test error")
        ax_plot.set_xscale("log")
        ax_plot.set_title(r"Bias and variance against $\lambda$", fontsize=11.5)
        ax_plot.set_xlabel(xlabel, fontsize=11)
        ax_plot.set_ylabel(ylabel, fontsize=11)
        if ylim is not None:
            ax_plot.set_ylim(ylim)
        ax_plot.legend(loc="upper center", fontsize=9.0)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_ensemble_fits(
        cls,
        x_dense: Any,
        h_dense: Any,
        ensembles: Dict[str, Tuple[Any, Any]],
        n_show: int = 20,
        figsize: Tuple[float, float] = (11, 3.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """One panel per entry {title: (fits of the L data sets, their average)}, all pre-computed on x_dense."""
        fig, axes = plt.subplots(1, len(ensembles), figsize=figsize, sharey=True, squeeze=False)
        x_ev, h_ev = _to_numpy(x_dense).ravel(), _to_numpy(h_dense).ravel()
        for ax, (title, (preds_all, mean_pred)) in zip(axes[0], ensembles.items()):
            for l in range(min(n_show, len(preds_all))):
                ax.plot(x_ev, _to_numpy(preds_all[l]).ravel(), color="firebrick", alpha=0.3, lw=0.9)
            ax.plot(x_ev, _to_numpy(mean_pred).ravel(), color="darkred", lw=2.4, label=r"average $\mathbb{E}_\mathcal{D}[f(x; \mathcal{D})]$")
            ax.plot(x_ev, h_ev, color="forestgreen", lw=2.0, linestyle="--", label=r"$h(x)$")
            ax.set_title(title, fontsize=10.0)
            ax.set_xlabel("$x$")
            ax.set_ylim(-1.6, 1.6)
            ax.grid(True)
        axes[0][0].set_ylabel(r"$f(x; \mathcal{D})$")
        axes[0][0].legend(loc="lower left", fontsize=8.0)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_least_squares_geometry(
        cls,
        Phi: Any,
        t: Any,
        y: Any,
        w: Any,
        figsize: Tuple[float, float] = (7.5, 5.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Draw t, its projection y = Phi w and the residual t - y over the plane spanned by the two columns of Phi (N = 3); receives pre-computed y and w."""
        Phi_np = _to_numpy(Phi)
        t_np = _to_numpy(t, ravel=True)
        y_np = _to_numpy(y, ravel=True)
        col_0, col_1 = Phi_np[:, 0], Phi_np[:, 1]

        # Patch of the subspace S: coefficients around the columns and around y = Phi w
        coef_y = _to_numpy(w, ravel=True)
        a_lo, a_hi = min(0.0, coef_y[0]) - 0.2, max(1.0, coef_y[0]) + 0.2
        b_lo, b_hi = min(0.0, coef_y[1]) - 0.2, max(1.0, coef_y[1]) + 0.2
        A, B = np.meshgrid(np.linspace(a_lo, a_hi, 2), np.linspace(b_lo, b_hi, 2))
        plane = A[..., None] * col_0 + B[..., None] * col_1

        # Camera 35 degrees above S, rolled so that the normal of S (direction of e) points up on screen
        n = np.cross(col_0, col_1)
        n = n / np.linalg.norm(n) * np.sign(n @ (t_np - y_np))
        u = np.cross(n, y_np) / np.linalg.norm(np.cross(n, y_np))
        d = np.cos(np.radians(55)) * n + np.sin(np.radians(55)) * u
        z_p = np.array([0.0, 0.0, 1.0]) - d[2] * d
        n_p = n - (n @ d) * d
        roll = np.degrees(np.arctan2(np.cross(z_p, n_p) @ d, z_p @ n_p))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(np.degrees(np.arcsin(d[2])), np.degrees(np.arctan2(d[1], d[0])), -roll)
        ax.plot_surface(plane[..., 0], plane[..., 1], plane[..., 2], color="lightsteelblue", alpha=0.35,
                        edgecolor="steelblue", linewidth=0.8)
        ax.text(*(plane[1, 1] * 0.92), r"$\mathcal{S}$", fontsize=16, color="steelblue")

        origin = np.zeros(3)
        arrows = [
            (col_0, "dimgray", r"$\boldsymbol{\varphi}_0$"),
            (col_1, "dimgray", r"$\boldsymbol{\varphi}_1$"),
            (t_np, "royalblue", r"$\mathbf{t}$"),
            (y_np, "firebrick", r"$\mathbf{y}$"),
        ]
        lift = 0.06 * np.linalg.norm(t_np) * n  # labels slightly above S, off the arrows
        for vec, color, label in arrows:
            ax.quiver(*origin, *vec, color=color, lw=2.2, arrow_length_ratio=0.08)
            ax.text(*(vec * 1.05 + lift), label, fontsize=15, color=color)

        # Residual e = t - y as a dashed segment from y to t
        ax.plot(*np.stack([y_np, t_np]).T, color="black", lw=1.8, linestyle="--")
        # The camera looks along u, so y runs horizontally on screen: put the label left of the dashed line
        ax.text(*((y_np + t_np) / 2 + 0.04 * y_np), r"$\mathbf{e} = \mathbf{t} - \mathbf{y}$", fontsize=13, ha="right")

        # Right-angle marker at y between e and the direction of y within S
        size = 0.12 * min(np.linalg.norm(t_np - y_np), np.linalg.norm(y_np))
        u_e = (t_np - y_np) / np.linalg.norm(t_np - y_np)
        u_s = -y_np / np.linalg.norm(y_np)
        corner = np.stack([y_np + size * u_e, y_np + size * (u_e + u_s), y_np + size * u_s])
        ax.plot(*corner.T, color="black", lw=1.2)

        ax.set_axis_off()
        ax.set_title(r"Least squares in $\mathbb{R}^N$ ($N = 3$, $M = 2$)", fontsize=12)
        # Equal scale on all axes, so that the right angle looks like one
        pts = np.vstack([plane.reshape(-1, 3), t_np, y_np, origin])
        lo, hi = pts.min(0), pts.max(0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 3. Classification & Decision Boundaries (05_classification, 06_deep_networks)
# ---------------------------------------------------------------------------

class DeepClassificationPlotter(BasePlotter):
    """Encapsulated decision boundary visualizers for single-layer and deep classifiers."""

    @classmethod
    def plot_hyperplane_geometry(
        cls,
        w: Any,
        w0: float,
        x: Any,
        x_perp: Any,
        xlim: Tuple[float, float] = (-1.0, 3.5),
        ylim: Tuple[float, float] = (-1.0, 3.5),
        figsize: Tuple[float, float] = (6.0, 5.2),
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Decision boundary w^T x + w0 = 0 in 2D with the normal w, a point x, its projection x_perp, the signed
        distance r and the distance -w0/||w|| of the boundary from the origin (after Fig. 5.1); values pre-computed."""
        fig, ax = cls.create_canvas(figsize=figsize)
        w_np, x_np, xp_np = (cls.to_numpy(v, ravel=True).astype(float) for v in (w, x, x_perp))
        norm = float(np.linalg.norm(w_np))
        x1 = np.linspace(*xlim, 200)
        ax.plot(x1, -(w0 + w_np[0] * x1) / w_np[1], color="firebrick", lw=2.2, label=r"$y(\mathbf{x}) = 0$")
        foot = -w0 / norm**2 * w_np                              # point of the boundary closest to the origin
        unit_w = w_np / norm
        along = np.array([-unit_w[1], unit_w[0]])                # direction within the boundary
        side = 1.0 if along @ (xp_np - foot) > 0 else -1.0       # side of foot on which x_perp lies
        # Distance of the boundary from the origin; its label sits beyond the origin, away from the boundary
        ax.annotate("", xy=foot, xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="gray", lw=1.4))
        away = -foot / max(float(np.linalg.norm(foot)), 1e-12)
        ax.text(*(0.12 * away), r"$-w_0 / \|\mathbf{w}\|$", color="gray", fontsize=11,
                ha="right" if away[0] < 0 else "left", va="top" if away[1] < 0 else "bottom")
        # The normal vector, drawn from a boundary point on the other side of the foot, away from x and x_perp
        start = foot - side * 1.2 * along
        ax.annotate("", xy=start + unit_w, xytext=start, arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
        ax.text(*(start + unit_w + 0.08 * unit_w), r"$\mathbf{w}$", fontsize=13, ha="left", va="bottom")
        ax.plot(*x_np, "o", color="royalblue", ms=8)
        ax.text(*(x_np + np.array([0.08, 0.08])), r"$\mathbf{x}$", fontsize=13, color="royalblue")
        ax.plot(*xp_np, "o", color="black", ms=6)
        ax.text(*(xp_np + np.array([-0.35, -0.3])), r"$\mathbf{x}_\perp$", fontsize=13)
        ax.plot([xp_np[0], x_np[0]], [xp_np[1], x_np[1]], color="royalblue", lw=1.6, linestyle="--")
        ax.text(*((xp_np + x_np) / 2 + np.array([0.08, -0.15])), r"$r$", fontsize=13, color="royalblue")
        ax.plot([0], [0], "k+", ms=10)
        ax.text(xlim[1] - 1.2, ylim[1] - 0.5, r"$\mathcal{R}_1$: $y > 0$", fontsize=12)
        ax.text(xlim[0] + 0.2, ylim[0] + 0.3, r"$\mathcal{R}_2$: $y < 0$", fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc="lower right", fontsize=9.5)
        return cls.finalize(fig, ax=ax, show=show)

    @classmethod
    def plot_reject_option(
        cls,
        x: Any,
        posteriors: Dict[str, Any],
        theta: float,
        reject: Any,
        figsize: Tuple[float, float] = (7.5, 3.8),
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Posterior curves {label: p(C_k | x)} over x, the threshold theta and the pre-computed reject region
        (boolean mask over x) shaded (after Fig. 5.7)."""
        fig, ax = cls.create_canvas(figsize=figsize)
        x_np, mask = cls.to_numpy(x, ravel=True), cls.to_numpy(reject, ravel=True).astype(bool)
        for (label, p), col in zip(posteriors.items(), ["royalblue", "firebrick", "forestgreen"]):
            ax.plot(x_np, cls.to_numpy(p, ravel=True), color=col, lw=2.2, label=label)
        ax.axhline(theta, color="black", linestyle="--", lw=1.2, label=rf"$\theta = {theta:g}$")
        ax.fill_between(x_np, 0, 1, where=mask, color="gray", alpha=0.25, label="reject region")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$p(\mathcal{C}_k \mid x)$")
        ax.legend(loc="center right", fontsize=9)
        return cls.finalize(fig, ax=ax, show=show)


# ---------------------------------------------------------------------------
# 4. Optimization Landscapes & Convergence Dynamics (07_gradient_descent, 08_backpropagation)
# ---------------------------------------------------------------------------

class DeepOptimizationPlotter(BasePlotter):
    """Optimization paths on pre-computed error functions (07_gradient_descent)."""

    @classmethod
    def plot_descent_1d(
        cls,
        w_grid: Any,
        E_grid: Any,
        trace_w: Any,
        trace_E: Any,
        title: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (7.5, 4.0),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Error function E(w) pre-computed on a grid and the iterates (w, E(w)) of a descent method, joined in order."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        ax_plot.plot(cls.to_numpy(w_grid, ravel=True), cls.to_numpy(E_grid, ravel=True), color="royalblue", lw=2.2,
                     label="$E(w)$")
        tw, tE = cls.to_numpy(trace_w, ravel=True), cls.to_numpy(trace_E, ravel=True)
        ax_plot.plot(tw, tE, "o-", color="firebrick", ms=5, lw=1.2, label="iterates $w^{(\\tau)}$")
        ax_plot.plot(tw[:1], tE[:1], "s", color="black", ms=7, label="$w^{(0)}$")
        if ylim is not None:
            ax_plot.set_ylim(*ylim)
        ax_plot.set_title(title or "", fontsize=11.5)
        ax_plot.set_xlabel("$w$")
        ax_plot.set_ylabel("$E(w)$")
        ax_plot.legend(loc="best", fontsize=9.5)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_contour_trajectories(
        cls,
        xx: Any,
        yy: Any,
        E_grid: Any,
        trajectories: Optional[Dict[str, Any]] = None,
        directions: Optional[Dict[str, Any]] = None,
        minimum: Optional[Sequence[float]] = None,
        curves: Optional[Dict[str, Any]] = None,
        points: Optional[Dict[str, Sequence[float]]] = None,
        show_minimum: bool = True,
        minimum_label: str = "minimum",
        equal_aspect: bool = False,
        legend_loc: str = "upper right",
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.5, 4.6),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Contours of an error function pre-computed on a grid, optimization paths {label: (T, 2) iterates},
        optional direction arrows {label: 2-vector} drawn from the minimum, plain curves {label: (T, 2)} (e.g.
        constraint boundaries) and marked points {label: (x, y)}."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax, grid=False)
        xx, yy, Z = cls.to_numpy(xx), cls.to_numpy(yy), cls.to_numpy(E_grid)
        levels = np.quantile(Z, np.linspace(0.01, 0.95, 14))
        ax_plot.contour(xx, yy, Z, levels=np.unique(levels), cmap="viridis", alpha=0.6)
        for (label, path), col in zip((trajectories or {}).items(), ["firebrick", "royalblue", "darkorange",
                                                                      "forestgreen", "purple"]):
            P = cls.to_numpy(path).reshape(-1, 2)
            ax_plot.plot(P[:, 0], P[:, 1], "o-", color=col, ms=3.5, lw=1.4, label=label)
        for (label, curve), col in zip((curves or {}).items(), ["crimson", "darkorange", "purple"]):
            C = cls.to_numpy(curve).reshape(-1, 2)
            ax_plot.plot(C[:, 0], C[:, 1], "-", color=col, lw=2.2, label=label)
        for (label, point), marker in zip((points or {}).items(), ["o", "s", "D", "^"]):
            ax_plot.plot(*np.asarray(point, dtype=float), marker, color="black", ms=8, linestyle="none", label=label)
        origin = np.zeros(2) if minimum is None else np.asarray(minimum, dtype=float)
        if show_minimum:
            ax_plot.plot(*origin, "k*", ms=12, label=minimum_label)
        for label, v in (directions or {}).items():
            v = cls.to_numpy(v, ravel=True).astype(float)
            ax_plot.annotate("", xy=origin + v, xytext=origin, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8))
            ax_plot.text(*(origin + 1.1 * v), label, fontsize=12, ha="center", va="center")
        ax_plot.set_xlim(float(np.min(xx)), float(np.max(xx)))
        ax_plot.set_ylim(float(np.min(yy)), float(np.max(yy)))
        if equal_aspect:
            ax_plot.set_aspect("equal")
        ax_plot.set_xlabel("$w_1$")
        ax_plot.set_ylabel("$w_2$")
        ax_plot.set_title(title or "", fontsize=11.5)
        ax_plot.legend(loc=legend_loc, fontsize=9)
        return cls.finalize(fig, ax=ax_plot, show=show)


class ComputationalGraphPlotter(BasePlotter):
    """Computational graphs with pre-computed forward values and adjoints (08_backpropagation)."""

    @classmethod
    def plot_computational_graph(
        cls,
        nodes: Dict[str, Tuple[float, float, str]],
        edges: Sequence[Tuple[str, str]],
        values: Optional[Dict[str, str]] = None,
        adjoints: Optional[Dict[str, str]] = None,
        edge_labels: Optional[Dict[Tuple[str, str], str]] = None,
        title: Optional[str] = None,
        radius: float = 0.32,
        figsize: Tuple[float, float] = (10.0, 4.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Nodes {name: (x, y, label)} and directed edges; forward values (black, above) and adjoints (red,
        below) are pre-computed strings, so the figure shows exactly the numbers of the notebook."""
        fig, ax = plt.subplots(figsize=figsize)
        for u, v in edges:
            (x0, y0, _), (x1, y1, _) = nodes[u], nodes[v]
            d = np.hypot(x1 - x0, y1 - y0)
            start = (x0 + radius * (x1 - x0) / d, y0 + radius * (y1 - y0) / d)
            end = (x1 - radius * (x1 - x0) / d, y1 - radius * (y1 - y0) / d)
            ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.4))
            if edge_labels and (u, v) in edge_labels:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, edge_labels[(u, v)], ha="center", va="bottom",
                        fontsize=10, color="#33415c")
        for name, (x, y, label) in nodes.items():
            ax.add_patch(plt.Circle((x, y), radius, facecolor="#dbe7f5", edgecolor="#33415c", lw=1.6, zorder=3))
            ax.text(x, y, label, ha="center", va="center", fontsize=12, zorder=4)
            if values and name in values:
                ax.text(x, y + radius + 0.08, values[name], ha="center", va="bottom", fontsize=10.5, zorder=4)
            if adjoints and name in adjoints:
                ax.text(x, y - radius - 0.08, adjoints[name], ha="center", va="top", fontsize=10.5,
                        color="firebrick", zorder=4)
        xs = [x for x, _, _ in nodes.values()]
        ys = [y for _, y, _ in nodes.values()]
        ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12)
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 5. Deep Diagnostics, Convolutions, Sequences & Graphs
# ---------------------------------------------------------------------------

class DeepArchitecturePlotter(BasePlotter):
    """Encapsulated diagnostics for activations, feature maps, attention, sequences, and graphs."""

    @classmethod
    def plot_boxes(
        cls,
        image: Any,
        boxes: Dict[str, Any],
        scores: Optional[Dict[str, Sequence[float]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5.0, 5.0),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """An image with pre-computed bounding boxes {label: (n, 4) corners (x0, y0, x1, y1)}, one colour per label,
        and optional scores written at the boxes."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax, grid=False)
        ax_plot.imshow(cls.to_numpy(image), cmap="gray")
        for (label, group), col in zip(boxes.items(), ["royalblue", "firebrick", "forestgreen", "darkorange"]):
            for k, (x0, y0, x1, y1) in enumerate(cls.to_numpy(group).reshape(-1, 4)):
                ax_plot.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=col, lw=2,
                                                label=label if k == 0 else None))
                if scores is not None and label in scores:
                    ax_plot.text(x0, y0 - 1, f"{scores[label][k]:.2f}", color=col, fontsize=9, va="bottom")
        ax_plot.axis("off")
        ax_plot.set_title(title or "", fontsize=11)
        ax_plot.legend(loc="lower right", fontsize=9)
        return cls.finalize(fig, ax=ax_plot, show=show)

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
plot_conditional_density = DeepRegressionPlotter.plot_conditional_density
plot_polynomial_extrapolation = DeepRegressionPlotter.plot_polynomial_extrapolation
plot_dataset_size_remedy = DeepRegressionPlotter.plot_dataset_size_remedy
plot_bias_variance_curve = DeepRegressionPlotter.plot_bias_variance_curve
plot_ensemble_fits = DeepRegressionPlotter.plot_ensemble_fits
plot_least_squares_geometry = DeepRegressionPlotter.plot_least_squares_geometry

# Classification Facades
plot_hyperplane_geometry = DeepClassificationPlotter.plot_hyperplane_geometry
plot_reject_option = DeepClassificationPlotter.plot_reject_option

# Optimization Facades
plot_descent_1d = DeepOptimizationPlotter.plot_descent_1d
plot_computational_graph = ComputationalGraphPlotter.plot_computational_graph
plot_contour_trajectories = DeepOptimizationPlotter.plot_contour_trajectories

# Architecture & Diagnostics Facades
plot_boxes = DeepArchitecturePlotter.plot_boxes
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

