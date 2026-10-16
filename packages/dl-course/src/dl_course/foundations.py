"""
Modular helper utilities for Mathematical Foundations (ML-DL / 01_foundations).

Provides structured, decoupled visualization classes and functional facades for:
1. Probability Distributions & Empirical Diagnostics (02_probability)
2. Information Theory Visualizations (03_information)
3. Linear Algebra, Vector Geometry & Dimensionality Reduction (04_algebra)
4. Mathematical Functions & Activation Profiles (05_functions)
5. Model Diagnostics & Quality Metrics (06_mlops)
"""

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import torch


# ---------------------------------------------------------------------------
# 1. Core Shared Foundation (Asset Resolution, Aesthetics, Base Plotters)
# ---------------------------------------------------------------------------

from .shared import (  # noqa: F401 -- re-exported for notebooks as hp.<name>
    COURSE_RAW_URL,
    DATA_BASE_URL,
    BasePlotter,
    compress_image_svd,
    get_data_path,
    get_file,
    get_url,
    plot_correlation_matrix,
    plot_fitted_curves,
    plot_pairgrid,
    plot_image,
    print_math,
    setup_theme,
)




# ---------------------------------------------------------------------------
# 2. Probability & Empirical Diagnostics Plotter (02_probability)
# ---------------------------------------------------------------------------

class ProbabilityPlotter(BasePlotter):
    """Visualizer for empirical probability distributions and diagnostics."""

    @classmethod
    def plot_sample_histograms(
        cls,
        samples: Dict[str, Any],
        title: Optional[str] = None,
        bins: int = 80,
        xlabel: str = "$x$",
        figsize: Tuple[float, float] = (8.0, 4.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ):
        """Overlay density histograms of named, pre-drawn samples."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel="density")
        for label, values in samples.items():
            ax.hist(cls.to_numpy(values, ravel=True), bins=bins, density=True, alpha=0.45, label=label)
        ax.legend(loc="best", fontsize=10)
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_gmm_samples(
        cls,
        X_gmm: Any,
        title: str = "Samples from a Gaussian Mixture Model",
        figsize: Tuple[float, float] = (6.0, 5.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ):
        """Scatter-plot pre-sampled 2D points from a Gaussian mixture model."""
        X_np = cls.to_numpy(X_gmm)
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel="$x_1$", ylabel="$x_2$")
        ax.scatter(X_np[:, 0], X_np[:, 1], color="royalblue", alpha=0.7, edgecolors="k", s=35)
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_joint_distribution_heatmap(
        cls,
        joint_matrix: np.ndarray,
        x_labels: Optional[Sequence[str]] = None,
        y_labels: Optional[Sequence[str]] = None,
        title: str = r"Joint Probability Distribution $p(x, y)$",
        figsize: Tuple[float, float] = (6.0, 4.5),
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """Render 2D joint discrete probability distribution matrix as an annotated heatmap."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        sns.heatmap(
            joint_matrix,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=x_labels if x_labels is not None else True,
            yticklabels=y_labels if y_labels is not None else True,
            cbar=True,
            ax=ax,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("$y$", fontsize=11)
        ax.set_ylabel("$x$", fontsize=11)
        fig.tight_layout()
        plt.show()
        return fig, ax


# ---------------------------------------------------------------------------
# 3. Vector Geometry & Optimization Trace Plotter (04_algebra)
# ---------------------------------------------------------------------------

class VectorGeometryPlotter(BasePlotter):
    """Visualizer for 2D vectors, rotations, and optimization traces."""

    @classmethod
    def plot2d(
        cls,
        tensor: torch.Tensor,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5.0, 5.0),
    ) -> None:
        """Plot 2D vectors emanating from the origin using Matplotlib quiver."""
        plt.figure(figsize=figsize)
        t_np = cls.to_numpy(tensor)
        o = np.zeros_like(t_np)
        plt.quiver(*o, *t_np, angles="xy", scale_units="xy", scale=1, color=["r", "g", "b", "m"])
        mx = float(np.max(np.abs(t_np))) + 0.5
        plt.xlim(-mx, mx)
        plt.ylim(-mx, mx)
        if title:
            plt.title(title, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    @classmethod
    def scale_vector(cls, t_vec: torch.Tensor, alpha: float = 1.0) -> None:
        """Scale a vector and render it with plot2d."""
        if alpha != 0:
            cls.plot2d(torch.cat((t_vec, t_vec * alpha), dim=1), title=f"Scale factor: alpha={alpha}")
        else:
            cls.plot2d(t_vec, title="Scale factor: alpha=0")

    @classmethod
    def inspect_angle(cls, y1: float = 0.0, y2: float = 1.0) -> None:
        """Inspect angle and inner product between (1, 0) and (y1, y2)."""
        x_base = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        y_comp = torch.tensor([[float(y1)], [float(y2)]], dtype=torch.float32)
        cos_val = cls.to_scalar((torch.matmul(x_base.t(), y_comp) / (torch.norm(x_base) * torch.norm(y_comp))).clamp(-1.0, 1.0))
        deg = math.degrees(math.acos(cos_val))
        dot_val = cls.to_scalar(torch.matmul(x_base.t(), y_comp))
        print_math(
            rf"Inner product $\mathbf{{x}}^\mathrm{{T}} \mathbf{{y}} = {dot_val:.3f}$ "
            rf"| Cosine $\cos \theta = {cos_val:.3f}$ | Angle $\theta = {deg:.1f}^\circ$"
        )
        cls.plot2d(torch.cat((x_base, y_comp), dim=1), title=f"Angle: {deg:.1f} deg")

    @classmethod
    def rotate_vector(cls, alpha: float = 90.0) -> None:
        """Rotate base vector (1, 0) by angle alpha degrees."""
        rad = math.radians(alpha)
        rot_mat = torch.tensor([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=torch.float32)
        v_init = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        v_rot = torch.matmul(rot_mat, v_init)
        v_0 = cls.to_scalar(v_rot[0, 0])
        v_1 = cls.to_scalar(v_rot[1, 0])
        print_math(
            rf"Rotation angle $\alpha = {alpha:.1f}^\circ$: rotated vector "
            rf"$\mathbf{{v}}_\mathrm{{rot}} = ({v_0:.3f}, {v_1:.3f})^\mathrm{{T}}$"
        )
        cls.plot2d(torch.cat((v_rot, v_init), dim=1), title=f"Rotation by {alpha} deg")

    @classmethod
    def plot_out(
        cls,
        out: Sequence[Any],
        title: str = "Gradient Descent Optimization Trace",
        figsize: Tuple[float, float] = (7.0, 4.0),
    ) -> None:
        """Render gradient descent optimization history (log-scale gradient norm)."""
        out_vals = [cls.to_scalar(x) for x in out]
        if len(out_vals) > 0:
            print_math(
                rf"Finished optimization in {len(out_vals)} iterations with final gradient norm "
                rf"$\|\nabla E(\mathbf{{w}})\| = {out_vals[-1]:.4e}$"
            )
        fig, ax = cls.canvas(figsize=figsize, title=title, xlabel=r"Iteration $k$", ylabel=r"Gradient Norm $\|\nabla E(\mathbf{w})\|_2$")
        ax.plot(range(1, len(out_vals) + 1), out_vals, color="royalblue", lw=2)
        ax.set_yscale("log")
        cls.finalize(fig, ax, show=True)


# ---------------------------------------------------------------------------
# 4. Dimensionality Reduction & PCA Plotter (04_algebra)
# ---------------------------------------------------------------------------

class PCAPlotter(BasePlotter):
    """Visualizer for PCA projections, biplots, and variance decomposition."""

    @classmethod
    def plot_scores(
        cls,
        X_us: torch.Tensor,
        us_columns: Sequence[str],
        x: float = 0.0,
        y: float = 0.0,
    ) -> None:
        """Plot 2D PCA orthogonal projection and orientation lines."""
        ort = torch.tensor([[float(x)], [float(y)]], dtype=torch.float32)
        fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)
        ax[0].set_ylabel(us_columns[1], fontsize=11)
        X_plot = cls.to_numpy(X_us)

        for a in ax:
            a.set_xlabel(us_columns[0], fontsize=11)
            a.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.7)
            l, r = a.get_xlim()
            ort_0 = cls.to_scalar(ort[0])
            ort_1 = cls.to_scalar(ort[1])
            if abs(ort_0) > 1e-6:
                x_vals = np.linspace(l, r, 20)
                y_vals = (ort_1 / ort_0) * x_vals
                a.plot(x_vals, y_vals, c="orange", lw=2)
            a.grid(True, linestyle="--", alpha=0.5)

        denom = cls.to_scalar(torch.matmul(ort.t(), ort))
        if abs(denom) > 1e-6:
            proj_matrix = torch.matmul(ort, ort.t()) / denom
            X_sub = X_us[:, :2] if hasattr(X_us, "device") else torch.tensor(X_us[:, :2], dtype=torch.float32)
            projected = cls.to_numpy(torch.matmul(X_sub, proj_matrix))
            for i in range(len(X_plot)):
                ax[1].plot([X_plot[i, 0], projected[i, 0]], [X_plot[i, 1], projected[i, 1]], c="green", alpha=0.4)
            ax[1].scatter(projected[:, 0], projected[:, 1], marker="x", c="k", s=25)

        fig.tight_layout()
        plt.show()

    @classmethod
    def biplot(
        cls,
        z1: int,
        z2: int,
        sc: Any,
        comps: Any,
        obs: Sequence[Any],
        features: Sequence[str],
        colors: Sequence[str],
    ) -> None:
        """Render PCA biplot with observation scores and loading vectors."""
        x = cls.to_numpy(sc)[:, z1]
        y = cls.to_numpy(sc)[:, z2]

        fig, ax = cls.canvas(figsize=(9.0, 8.0), xlabel=f"$z_{{{z1+1}}}$", ylabel=f"$z_{{{z2+1}}}$")
        sx = float((x.max() - x.min()) / 2)
        sy = float((y.max() - y.min()) / 2)

        ax.scatter(x, y, alpha=0.6)
        for i in range(len(obs)):
            ax.text(x[i], y[i], str(obs[i]), ha="center", fontsize=9, alpha=0.8)

        comps_mat = cls.to_numpy(comps)[[z1, z2], :].T
        for i in range(len(comps_mat)):
            ax.arrow(
                0,
                0,
                comps_mat[i, 0] * sx,
                comps_mat[i, 1] * sy,
                ec=colors[i % len(colors)],
                head_width=0.1,
                head_length=0.1,
                fc=colors[i % len(colors)],
                lw=2,
            )
            ax.text(
                comps_mat[i, 0] * sx * 1.15,
                comps_mat[i, 1] * sy * 1.15,
                features[i],
                color=colors[i % len(colors)],
                fontsize=12,
                weight="bold",
            )
        cls.finalize(fig, ax, show=True)

    @classmethod
    def plot_pca_vectors(cls, X: torch.Tensor, feature_names: Sequence[str], loadings: torch.Tensor) -> None:
        """Plot 2D feature scatter with PCA principal component direction vectors."""
        X_plot = cls.to_numpy(X)
        loadings_np = cls.to_numpy(loadings)
        colors = ["red", "orange"]

        fig, ax = cls.canvas(figsize=(6.0, 5.0), title="Data Scatter and Principal Directions", xlabel=feature_names[0], ylabel=feature_names[1])
        ax.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.7)
        for i in range(min(2, len(loadings_np))):
            ax.arrow(
                0,
                0,
                loadings_np[i, 0],
                loadings_np[i, 1],
                ec=colors[i % len(colors)],
                head_width=0.12,
                head_length=0.12,
                fc=colors[i % len(colors)],
                lw=2,
                label=f"Principal direction $\\mathbf{{u}}_{{{i+1}}}$",
            )
        ax.legend()
        cls.finalize(fig, ax, show=True)

    @classmethod
    def plot_loadings_heatmap(cls, loadings: torch.Tensor, feature_names: Sequence[str]) -> None:
        """Render heatmap of principal component loadings across feature dimensions."""
        loadings_np = cls.to_numpy(loadings)
        fig, ax = cls.canvas(figsize=(7.0, 4.0), title="Principal Component Directions (Loadings)", grid=False)
        sns.heatmap(
            loadings_np,
            cmap=sns.diverging_palette(10, 240, as_cmap=True),
            annot=True,
            fmt=".2f",
            xticklabels=feature_names,
            yticklabels=[f"u_{i+1}" for i in range(len(loadings_np))],
            ax=ax,
        )
        cls.finalize(fig, ax, show=True)

    @classmethod
    def plot_explained_variance(cls, ve: torch.Tensor) -> None:
        """Render 2-panel scree plot: individual variance and cumulative variance ratio."""
        ve_np = cls.to_numpy(ve)
        ve_ratio = ve_np / ve_np.sum() if ve_np.sum() > 0 else ve_np
        cum_ratio = np.cumsum(ve_ratio)

        k_range = list(range(1, len(ve_np) + 1))

        fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
        ax[0].bar(k_range, ve_np, color="royalblue")
        ax[0].set_ylabel("Variance Explained")
        ax[0].set_xlabel("Principal Component")
        ax[0].set_title("Individual Variance")
        ax[0].set_xticks(k_range)
        ax[0].grid(True, linestyle="--", alpha=0.5)

        ax[1].bar(k_range, cum_ratio, color="forestgreen")
        ax[1].set_ylabel("Cumulative Ratio")
        ax[1].set_xlabel("Principal Component")
        ax[1].set_title("Cumulative Variance Ratio")
        ax[1].set_xticks(k_range)
        ax[1].grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        plt.show()

    @classmethod
    def plot_iris_pairplot(cls, df: pd.DataFrame) -> None:
        """Render pairplot of Iris features colored by species with lower KDE contours."""
        pair_grid = sns.pairplot(df, hue="species")
        pair_grid.map_lower(sns.kdeplot, levels=4, color=".2")
        plt.show()


# ---------------------------------------------------------------------------
# 5. Mathematical Functions & Activation Profiles (05_functions, 06_mlops)
# ---------------------------------------------------------------------------

class FunctionPlotter(BasePlotter):
    """Visualizer for activation curves, derivatives, and categorical distributions."""

    @classmethod
    def plot_function_and_derivative(
        cls,
        x: Sequence[float],
        y: Sequence[float],
        dy: Optional[Sequence[float]] = None,
        title: Optional[str] = None,
        fn_label: str = "Function $h(a)$",
        dfn_label: str = "Derivative $h'(a)$",
        figsize: Tuple[float, float] = (7.5, 4.0),
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """Render activation function curve and its analytical or autograd derivative."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.plot(x, y, color="royalblue", lw=2.5, label=fn_label)
        if dy is not None:
            ax.plot(x, dy, color="firebrick", lw=2.0, linestyle="--", label=dfn_label)

        ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.6)
        if title:
            ax.set_title(title, fontsize=12)
        ax.set_xlabel("Activation input ($a$)", fontsize=11.5)
        ax.set_ylabel("Output", fontsize=11.5)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        plt.show()
        return fig, ax

    @classmethod
    def plot_multi_curve_comparison(
        cls,
        x: Sequence[float],
        curves_dict: Dict[str, Sequence[float]],
        title: str = "Activation Function Comparison",
        xlabel: str = "Input ($a$)",
        ylabel: str = "Output $h(a)$",
        figsize: Tuple[float, float] = (8.0, 4.5),
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """Compare multiple activation functions across a shared input range."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        for name, y_vals in curves_dict.items():
            ax.plot(x, y_vals, lw=2.2, label=name)

        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="best", fontsize=10.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        plt.show()
        return fig, ax

    @classmethod
    def plot_curve_with_marker(
        cls,
        x_grid: Sequence[float],
        y_grid: Sequence[float],
        x_val: float,
        y_val: float,
        xlabel: str = "x",
        ylabel: str = "y",
        title: Optional[str] = None,
        curve_label: Optional[str] = None,
        marker_label: Optional[str] = None,
        vline_at: Optional[float] = None,
        figsize: Tuple[float, float] = (8.0, 4.2),
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """Render 1D mathematical curve with dynamic highlighted parameter marker."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.plot(x_grid, y_grid, color="crimson", lw=2.5, label=curve_label)
        ax.scatter([x_val], [y_val], color="royalblue", s=75, zorder=5, label=marker_label)
        if vline_at is not None:
            ax.axvline(vline_at, color="gray", linestyle="--", alpha=0.6)

        if title:
            ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="best", fontsize=10.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        plt.show()
        return fig, ax

    @classmethod
    def plot_categorical_probabilities(
        cls,
        categories: Sequence[str],
        probs_dict: Dict[str, Sequence[float]],
        title: str = "Softmax Temperature Output Distribution",
        figsize: Tuple[float, float] = (8.0, 4.0),
    ) -> None:
        """Compare categorical output distributions across softmax temperatures."""
        fig, ax = cls.canvas(figsize=figsize, title=title, ylabel="Probability $p_k$", grid=False)
        n_series = len(probs_dict)
        x = np.arange(len(categories))
        width = 0.8 / n_series

        for idx, (label, probs) in enumerate(probs_dict.items()):
            offset = (idx - n_series / 2 + 0.5) * width
            ax.bar(x + offset, probs, width, label=label, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=10.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        cls.finalize(fig, ax, show=True)

    @classmethod
    def plot_classifier_diagnostics(
        cls,
        cm: np.ndarray,
        y_prob: Optional[Sequence[float]] = None,
        class_names: Optional[Sequence[str]] = None,
        figsize: Tuple[float, float] = (10.0, 4.2),
    ) -> None:
        """
        Render 2-panel classification QA diagnostics: Confusion matrix and prediction distributions.
        Expects pre-computed confusion matrix cm from the calling notebook.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        labels = class_names if class_names is not None else ["Class 0", "Class 1"]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title("Confusion Matrix", fontsize=12)
        axes[0].set_xlabel(r"Predicted Label $\hat{t}$", fontsize=11)
        axes[0].set_ylabel(r"True Target $t$", fontsize=11)

        if y_prob is not None:
            sns.histplot(y_prob, bins=20, kde=True, ax=axes[1], color="royalblue")
            axes[1].axvline(0.5, color="crimson", linestyle="--", label="Decision Threshold ($p = 0.5$)")
            axes[1].set_title("Predicted Probability Distribution", fontsize=12)
            axes[1].set_xlabel(r"Predicted Probability $p(t=1 \mid \mathbf{x})$", fontsize=11)
            axes[1].set_ylabel("Count", fontsize=11)
            axes[1].legend(fontsize=10)
        else:
            axes[1].text(0.5, 0.5, "Probability scores not provided", ha="center", va="center")

        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 6. Public Functional Facades (100% Backward Compatibility)
# ---------------------------------------------------------------------------

plot_sample_histograms = ProbabilityPlotter.plot_sample_histograms
plot_gmm_samples = ProbabilityPlotter.plot_gmm_samples
plot_joint_distribution_heatmap = ProbabilityPlotter.plot_joint_distribution_heatmap

plot2d = VectorGeometryPlotter.plot2d
scale_vector = VectorGeometryPlotter.scale_vector
inspect_angle = VectorGeometryPlotter.inspect_angle
rotate_vector = VectorGeometryPlotter.rotate_vector
plot_out = VectorGeometryPlotter.plot_out
plot = plot_out
plot_convergence = plot_out

plot_scores = PCAPlotter.plot_scores
biplot = PCAPlotter.biplot
plot_pca_vectors = PCAPlotter.plot_pca_vectors
plot_loadings_heatmap = PCAPlotter.plot_loadings_heatmap
plot_explained_variance = PCAPlotter.plot_explained_variance
plot_iris_pairplot = PCAPlotter.plot_iris_pairplot

plot_function_and_derivative = FunctionPlotter.plot_function_and_derivative
plot_multi_curve_comparison = FunctionPlotter.plot_multi_curve_comparison
plot_curve_with_marker = FunctionPlotter.plot_curve_with_marker
plot_categorical_probabilities = FunctionPlotter.plot_categorical_probabilities
plot_classifier_diagnostics = FunctionPlotter.plot_classifier_diagnostics

# Conversion & Helper Facades
to_numpy = BasePlotter.to_numpy
to_scalar = BasePlotter.to_scalar
ensure_2d = BasePlotter.ensure_2d
eval_predictor = BasePlotter.eval_predictor
get_boundaries = BasePlotter.get_boundaries
create_2d_grid = BasePlotter.create_2d_grid

