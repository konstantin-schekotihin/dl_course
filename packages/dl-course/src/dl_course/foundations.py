"""
Modular helper utilities for Mathematical Foundations (ML-DL / 01_foundations).

Provides structured, decoupled visualization classes and functional facades for:
1. Probability Distributions & Empirical Diagnostics (02_probability)
2. Information Theory Visualizations (03_information)
3. Linear Algebra, Vector Geometry & Dimensionality Reduction (04_algebra)
4. Mathematical Functions & Activation Profiles (05_functions)
5. Model evaluation plots for 06_mlops come from shared (plot_confusion_matrix, plot_roc_curves)
"""

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
    get_data_path,
    get_file,
    get_url,
    plot_confusion_matrix,
    format_number,
    format_vector,
    plot_correlation_matrix,
    plot_directions,
    plot_explained_variance,
    plot_fitted_curves,
    plot_function_panels,
    plot_image,
    plot_image_grid,
    plot_loading_heatmap,
    plot_pairgrid,
    plot_pca_biplot,
    plot_projection,
    plot_roc_curves,
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
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Render 2D joint discrete probability distribution matrix as an annotated heatmap."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel="$y$", ylabel="$x$", grid=False)
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
        return cls.finalize(fig, ax, show=show)


# ---------------------------------------------------------------------------
# 3. Vector Geometry & Optimization Trace Plotter (04_algebra)
# ---------------------------------------------------------------------------

class VectorGeometryPlotter(BasePlotter):
    """Visualizer for 2D vectors and optimization traces."""

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
    def plot_convergence(
        cls,
        history: Sequence[float],
        title: Optional[str] = "Gradient Descent Convergence",
        ylabel: str = r"$\|\nabla E(\mathbf{w}^{(\tau)})\|$",
        figsize: Tuple[float, float] = (7.0, 4.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ):
        """Plot a pre-recorded optimization history (e.g. gradient norms) per iteration on a log scale."""
        values = cls.to_numpy(history, ravel=True)
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=r"Iteration $\tau$", ylabel=ylabel)
        ax.plot(range(1, len(values) + 1), values, color="royalblue", lw=2)
        ax.set_yscale("log")
        return cls.finalize(fig, ax, show=show)


# ---------------------------------------------------------------------------
# 4. Datasets for the PCA case studies (04_algebra); the PCA plots themselves are in shared.PCAPlotter
# ---------------------------------------------------------------------------

class IrisPlotter(BasePlotter):
    """Visualizer for the Iris dataset."""

    @classmethod
    def plot_iris_pairplot(cls, df: pd.DataFrame) -> None:
        """Render pairplot of Iris features colored by species with lower KDE contours."""
        pair_grid = sns.pairplot(df, hue="species")
        pair_grid.map_lower(sns.kdeplot, levels=4, color=".2")
        plt.show()


# ---------------------------------------------------------------------------
# 5. Mathematical Functions & Activation Profiles (03_information, 05_functions)
# ---------------------------------------------------------------------------

class FunctionPlotter(BasePlotter):
    """Visualizer for activation curves, derivatives, and categorical distributions."""

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
        hline_at: Optional[float] = None,
        figsize: Tuple[float, float] = (8.0, 4.2),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Render a pre-computed 1D curve with one highlighted point and optional vertical/horizontal guide lines."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        ax.plot(x_grid, y_grid, color="crimson", lw=2.5, label=curve_label)
        ax.scatter([x_val], [y_val], color="royalblue", s=75, zorder=5, label=marker_label)
        if vline_at is not None:
            ax.axvline(vline_at, color="gray", linestyle="--", alpha=0.6)
        if hline_at is not None:
            ax.axhline(hline_at, color="black", linestyle=":", alpha=0.7)
        ax.legend(loc="best", fontsize=10.0)
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_categorical_probabilities(
        cls,
        categories: Sequence[str],
        probs_dict: Dict[str, Any],
        title: str = "Softmax Temperature Output Distribution",
        ylabel: str = "Probability $y_k$",
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (8.0, 4.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Bar chart of pre-computed categorical distributions (e.g. softmax outputs), one bar group per series."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, ylabel=ylabel, ylim=ylim, grid=False)
        n_series = len(probs_dict)
        x = np.arange(len(categories))
        width = 0.8 / n_series

        for idx, (label, probs) in enumerate(probs_dict.items()):
            offset = (idx - n_series / 2 + 0.5) * width
            ax.bar(x + offset, cls.to_numpy(probs, ravel=True), width, label=label, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=10.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        return cls.finalize(fig, ax, show=show)


# ---------------------------------------------------------------------------
# 6. Public Functional Facades (100% Backward Compatibility)
# ---------------------------------------------------------------------------

plot_sample_histograms = ProbabilityPlotter.plot_sample_histograms
plot_gmm_samples = ProbabilityPlotter.plot_gmm_samples
plot_joint_distribution_heatmap = ProbabilityPlotter.plot_joint_distribution_heatmap

plot2d = VectorGeometryPlotter.plot2d
plot_convergence = VectorGeometryPlotter.plot_convergence

plot_iris_pairplot = IrisPlotter.plot_iris_pairplot

plot_curve_with_marker = FunctionPlotter.plot_curve_with_marker
plot_categorical_probabilities = FunctionPlotter.plot_categorical_probabilities

# Conversion & Helper Facades
to_numpy = BasePlotter.to_numpy
to_scalar = BasePlotter.to_scalar
ensure_2d = BasePlotter.ensure_2d
eval_predictor = BasePlotter.eval_predictor
get_boundaries = BasePlotter.get_boundaries
create_2d_grid = BasePlotter.create_2d_grid

