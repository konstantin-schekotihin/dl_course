"""
Modular class-based and functional helper utilities for AI and Machine Learning (AI-ML).
Provides structured, decoupled visualization and diagnostic routines for:
1. Course Aesthetics & Dataset Path Resolution
2. Supervised Regression & Exploratory Diagnostics (02_regression)
3. Instance-Based Classification & k-NN Surfaces (01_classifiers-kNN)
4. Logistic Regression & Sigmoidal Projections (03_logistic_regression)
5. Decision Surfaces, Margin Geometry & Ensembles (04_NN, 05_ensembles, 06_SVM)
6. Unsupervised Learning, PCA & Matrix Decompositions (07_unsupervised)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics


# ---------------------------------------------------------------------------
# 1. Core Shared Foundation (Asset Resolution, Aesthetics, Base Plotter)
# ---------------------------------------------------------------------------

from .shared import (  # noqa: F401 -- re-exported for notebooks as hp.<name>
    COURSE_RAW_URL,
    DATA_BASE_URL,
    BasePlotter,
    annotate,
    compress_image_svd,
    get_data_path,
    get_file,
    get_url,
    load_image,
    plot_binary,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_directions,
    plot_fitted_curves,
    plot_image,
    plot_loading_heatmap,
    plot_pairgrid,
    plot_pca_biplot,
    plot_roc_curves,
    print_math,
    setup_theme,
)

_to_numpy = BasePlotter.to_numpy  # short internal name used by the plotters below

cm_binary: ListedColormap = ListedColormap(["green", "blue"])



# ---------------------------------------------------------------------------
# 2. Supervised Regression & Response Projections (02_regression, 03_logistic_regression)
# ---------------------------------------------------------------------------

class AIMLRegressionPlotter(BasePlotter):
    """Encapsulated regression visualizers for linear fits, residuals, and logistic curves."""

    @classmethod
    def plot_residuals(
        cls,
        f: Union[Callable[[Any], Any], np.ndarray],
        pred: np.ndarray,
        resp: np.ndarray,
        show: bool = False,
        xlabel: str = "TV Budget ($x$)",
        ylabel: str = "Sales ($t$)",
        title: str = "Linear Regression Residuals",
        figsize: Tuple[float, float] = (8, 5.5),
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Axes]:
        """
        Plot regression observations, fitted model curve, and vertical residual drop-lines.
        Decoupled from ML error metric evaluation (metrics computed directly in notebook cells).
        """
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        pred_arr = cls.to_numpy(pred, ravel=True)
        resp_arr = cls.to_numpy(resp, ravel=True)
        est_arr = cls.eval_predictor(f, pred_arr.reshape(-1, 1), as_tensor_like=pred).ravel()

        x_min, x_max = float(pred_arr.min()), float(pred_arr.max())
        span = max(x_max - x_min, 1.0)
        l = np.linspace(x_min - 0.05 * span, x_max + 0.05 * span, num=500)

        ax_plot.vlines(pred_arr, resp_arr, est_arr, colors="royalblue", linestyles="dashed", alpha=0.5, label="Residuals")
        ax_plot.scatter(pred_arr, resp_arr, color="royalblue", alpha=0.6, s=40, label="Observations")
        ax_plot.scatter(pred_arr, est_arr, color="firebrick", s=30, label="Predictions")

        l_pred = cls.eval_predictor(f, l.reshape(-1, 1), as_tensor_like=pred).ravel()
        ax_plot.plot(l, l_pred, "firebrick", lw=2.2, label=r"Fitted $y(x, \mathbf{w})$")


        ax_plot.set_title(title, fontsize=12.5)
        ax_plot.set_xlabel(xlabel, fontsize=11.5)
        ax_plot.set_ylabel(ylabel, fontsize=11.5)
        ax_plot.legend(loc="upper left", fontsize=10.0)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=True)

    @classmethod
    def plot_linear_classification_fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Callable[[Any], Any],
        xlabel: str = r"Standardized Balance ($x$)",
        ylabel: str = r"Class Target ($t$)",
        title: str = "Linear Regression on Binary Classification",
        figsize: Tuple[float, float] = (8, 4.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Plot scatter observations and unbounded linear regression predictor."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        X_arr = _to_numpy(X).ravel()
        y_arr = _to_numpy(y).ravel()
        span = max(float(X_arr.max() - X_arr.min()), 1.0)
        l = np.linspace(X_arr.min() - 0.1 * span, X_arr.max() + 0.1 * span, num=200).reshape(-1, 1)

        ax_plot.scatter(X_arr, y_arr, alpha=0.5, color="royalblue", label="Observations ($x_n, t_n$)")
        ax_plot.plot(l, predict_fn(l), "firebrick", lw=2.2, label=r"Linear fit: $y(x, \mathbf{w}) = \mathbf{w}^\mathrm{T}\mathbf{x}$")
        ax_plot.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Threshold $y=0.5$")
        ax_plot.set_title(title, fontsize=12)
        ax_plot.set_xlabel(xlabel, fontsize=11.5)
        ax_plot.set_ylabel(ylabel, fontsize=11.5)
        ax_plot.legend(loc="upper left", fontsize=9.5)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_logistic_response_curve(
        cls,
        x_vals: Sequence[float],
        y_vals: Sequence[float],
        w_0: Optional[float] = None,
        w_1: Optional[float] = None,
        xlabel: str = r"Standardized Balance ($x$)",
        ylabel: str = r"Predicted Probability $p(t=1 \mid x)$",
        title: str = "Logistic Sigmoid Response Curve",
        figsize: Tuple[float, float] = (8, 4.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Plot logistic sigmoid response curve over feature range with decision threshold."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        if w_0 is not None and w_1 is not None:
            eq_label = rf"$\sigma({w_0:.1f} + {w_1:.1f}x)$"
        else:
            eq_label = r"$\sigma(a) = \frac{1}{1 + e^{-a}}$"

        ax_plot.plot(x_vals, y_vals, "royalblue", lw=2.4, label=eq_label)
        ax_plot.axhline(0.5, color="firebrick", linestyle="--", label="Decision threshold $p = 0.5$")
        ax_plot.set_title(title, fontsize=12)
        ax_plot.set_xlabel(xlabel, fontsize=11.5)
        ax_plot.set_ylabel(ylabel, fontsize=11.5)
        ax_plot.set_ylim(-0.05, 1.05)
        ax_plot.legend(loc="upper left", fontsize=10.0)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_positive_vs_log(
        cls,
        l: np.ndarray,
        f_vals: np.ndarray,
        log_vals: Optional[np.ndarray] = None,
        is_positive: bool = True,
        figsize: Tuple[float, float] = (8, 4.8),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Plot positive surrogate function f and its monotonic log transform."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        ax_plot.plot(l, f_vals, "royalblue", lw=2.2, label=r"Likelihood surrogate $f(x)$")
        if not is_positive:
            ax_plot.set_title(r"Warning: The function is not strictly positive ($f(x) \le 0$)", fontsize=12)
        elif log_vals is not None:
            ax_plot.plot(l, log_vals, "firebrick", lw=2.0, linestyle="--", label=r"Log-likelihood $\ln f(x)$")
            ax_plot.set_title(r"Monotonicity of the Logarithmic Transformation: $\arg\max f(x) = \arg\max \ln f(x)$", fontsize=11.5)
        ax_plot.set_xlabel("$x$", fontsize=11.5)
        ax_plot.set_ylabel("Value", fontsize=11.5)
        ax_plot.legend(loc="best", fontsize=10.0)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_curves(
        cls,
        x: Sequence[float],
        curves: Dict[str, Sequence[float]],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        logx: bool = False,
        mark_x: Optional[float] = None,
        mark_label: Optional[str] = None,
        figsize: Tuple[float, float] = (7.5, 4.2),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, Optional[plt.Axes]]]:
        """Draw pre-computed curves over a shared x axis (errors over flexibility, coefficient paths, losses)."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        x_arr = _to_numpy(x, ravel=True)
        marker = "o" if len(x_arr) <= 25 else None
        colors = plt.cm.tab20.colors if len(curves) > 10 else [None] * len(curves)  # distinct colours for many curves
        for (label, values), color in zip(curves.items(), colors):
            ax.plot(x_arr, _to_numpy(values, ravel=True), lw=2.0, marker=marker, ms=4, label=label, color=color)
        if logx:
            ax.set_xscale("log")
        if mark_x is not None:
            ax.axvline(mark_x, color="gray", linestyle="--", lw=1.2, label=mark_label)
        if len(curves) > 6:
            ax.legend(fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        else:
            ax.legend(fontsize=9.5, loc="best")
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_confounding(
        cls,
        x_mid: Sequence[float],
        rates: Dict[str, Sequence[float]],
        overall: Dict[str, float],
        values_by_group: Dict[str, Sequence[float]],
        xlabel: str,
        ylabel: str,
        value_label: str,
        figsize: Tuple[float, float] = (11.0, 4.0),
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, Optional[plt.Axes]]]:
        """Left: pre-computed rate per x-bin for each group (overall group rates dashed); right: boxplots of x per group."""
        fig, (ax_rate, ax_box) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]})
        colors = ["firebrick", "royalblue", "forestgreen", "darkorange"]
        for (group, values), color in zip(rates.items(), colors):
            ax_rate.plot(_to_numpy(x_mid, ravel=True), _to_numpy(values, ravel=True), color=color, lw=2, marker="o", ms=3, label=group)
            ax_rate.axhline(overall[group], color=color, linestyle="--", lw=1.2)
        ax_rate.set_xlabel(xlabel, fontsize=10.5)
        ax_rate.set_ylabel(ylabel, fontsize=10.5)
        ax_rate.legend(fontsize=9.5)
        ax_rate.grid(True, linestyle="--", alpha=0.5)
        ax_box.boxplot([_to_numpy(v, ravel=True) for v in values_by_group.values()])
        ax_box.set_xticks(range(1, len(values_by_group) + 1), list(values_by_group.keys()))
        ax_box.set_ylabel(value_label, fontsize=10.5)
        ax_box.grid(True, linestyle="--", alpha=0.5)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_residual_panels(
        cls,
        panels: Dict[str, Tuple[Sequence[float], Sequence[float]]],
        xlabel: str = r"Fitted value $\hat{t}_n$",
        ylabel: str = r"Residual $e_n$",
        band: Optional[float] = None,
        n_labels: int = 0,
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, Optional[plt.Axes]]]:
        """Scatter pre-computed residuals against fitted values (or leverage), one panel per entry; label the largest."""
        fig, axes = plt.subplots(1, len(panels), figsize=figsize or (5.2 * len(panels), 3.9), squeeze=False)
        for ax, (title, (x_vals, r_vals)) in zip(axes[0], panels.items()):
            x_arr, r_arr = _to_numpy(x_vals, ravel=True), _to_numpy(r_vals, ravel=True)
            ax.scatter(x_arr, r_arr, color="royalblue", alpha=0.55, s=16)
            ax.axhline(0, color="gray", lw=1.0)
            if band is not None:
                ax.axhline(band, color="firebrick", linestyle="--", lw=1.2)
                ax.axhline(-band, color="firebrick", linestyle="--", lw=1.2)
            for n in np.argsort(-np.abs(r_arr))[:n_labels]:
                ax.annotate(str(n), (x_arr[n], r_arr[n]), xytext=(4, 2), textcoords="offset points", fontsize=8.5, color="firebrick")
            ax.set_title(title, fontsize=11.5)
            ax.set_xlabel(xlabel, fontsize=10.5)
            ax.set_ylabel(ylabel, fontsize=10.5)
            ax.grid(True, linestyle="--", alpha=0.5)
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 3. Instance-Based Classification, Decision Surfaces & Margins (01, 04, 05, 06)
# ---------------------------------------------------------------------------

class AIMLClassificationPlotter(BasePlotter):
    """Encapsulated classification visualizers: k-NN, decision boundaries, SVM margins, confusion matrices."""

    @staticmethod
    def get_boundaries(
        X: np.ndarray, padding: float = 1.0
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute 2D coordinate plot boundaries with padding."""
        X_arr = _to_numpy(X)
        xlim = (float(np.min(X_arr[:, 0]) - padding), float(np.max(X_arr[:, 0]) + padding))
        ylim = (float(np.min(X_arr[:, 1]) - padding), float(np.max(X_arr[:, 1]) + padding))
        return xlim, ylim

    @classmethod
    def plot_knn_neighborhood_2d(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        query_pt: Tuple[float, float] = (5.5, 4.7),
        radius: float = 0.3,
        feature_names: Tuple[str, str] = ("Sepal Length ($x_1$)", "Petal Length ($x_2$)"),
        show_neighborhood: bool = False,
        figsize: Tuple[float, float] = (7.5, 6.5),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Render 2D scatter with decision query point and radius circle."""
        fig, ax_plot = cls.create_canvas(figsize=figsize, ax=ax)
        X_arr = _to_numpy(X)
        y_arr = _to_numpy(y)

        unique_classes = np.unique(y_arr)
        palette = sns.color_palette("deep", n_colors=len(unique_classes))

        for idx, cl_val in enumerate(unique_classes):
            mask = (y_arr == cl_val)
            ax_plot.scatter(
                X_arr[mask, 0],
                X_arr[mask, 1],
                color=palette[idx],
                label=f"Class: {cl_val}",
                edgecolors="w",
                s=50,
                alpha=0.85,
            )

        ax_plot.set_xlim(0, 8)
        ax_plot.set_ylim(0, 8)
        ax_plot.set_xlabel(feature_names[0], fontsize=12)
        ax_plot.set_ylabel(feature_names[1], fontsize=12)
        ax_plot.set_title("Instance-Based Learning: Nearest Neighbor Radius", fontsize=12.5)

        if show_neighborhood:
            ax_plot.scatter([query_pt[0]], [query_pt[1]], color="magenta", s=100, zorder=5, label=r"Query Point $\mathbf{x}_q$")
            circle = plt.Circle(query_pt, radius=radius, color="magenta", alpha=0.25, zorder=4)
            ax_plot.add_artist(circle)

        ax_plot.legend(loc="upper left", fontsize=10)
        ax_plot.grid(True)
        return cls.finalize(fig, ax=ax_plot, show=show)

    @classmethod
    def plot_classes_2d(
        cls,
        X: np.ndarray,
        t: np.ndarray,
        xlabel: str = "$x_1$",
        ylabel: str = "$x_2$",
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6.5, 4.6),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, Optional[plt.Axes]]]:
        """Scatter two-dimensional inputs X coloured by their class labels t."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        X_arr, t_arr = _to_numpy(X), _to_numpy(t, ravel=True)
        for label, color in zip(np.unique(t_arr), ["crimson", "royalblue", "forestgreen", "darkorange"]):
            mask = t_arr == label
            ax.scatter(X_arr[mask, 0], X_arr[mask, 1], c=color, alpha=0.7, edgecolors="k", s=28, label=f"class {label}")
        ax.legend(fontsize=9.5)
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_class_counts(
        cls,
        frames: Dict[str, pd.DataFrame],
        x: str,
        hue: str,
        n_cols: int = 2,
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, Optional[plt.Axes]]]:
        """Count plots of column x coloured by hue, one panel per (pre-split) data frame."""
        n_rows = (len(frames) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, sharey=True, squeeze=False, figsize=figsize or (6.0 * n_cols, 3.0 * n_rows))
        hue_order = sorted(pd.concat([df[hue] for df in frames.values()]).unique())
        for ax, (title, df) in zip(axes.ravel(), frames.items()):
            sns.countplot(x=x, data=df, hue=hue, hue_order=hue_order, ax=ax)
            ax.set_title(title, fontsize=11)
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.legend(fontsize=8)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_knn_classification_results(
        cls,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        test_y: np.ndarray,
        pred_y: np.ndarray,
        target_names: Optional[Sequence[str]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        feat_idx: Tuple[int, int] = (3, 0),
        figsize: Tuple[float, float] = (14, 5.5),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Dual-subplot evaluation for kNN: Left = Confusion Matrix; Right = 2D Scatter with Misclassification Markers."""
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        train_y_arr = _to_numpy(train_y)
        test_y_arr = _to_numpy(test_y)
        pred_y_arr = _to_numpy(pred_y)
        train_X_arr = _to_numpy(train_X)
        test_X_arr = _to_numpy(test_X)

        if target_names is None:
            target_names = np.unique(np.concatenate([train_y_arr, test_y_arr]))

        cm = metrics.confusion_matrix(test_y_arr, pred_y_arr, labels=target_names)
        sns.heatmap(
            cm,
            cmap="Blues",
            annot=True,
            fmt="d",
            ax=axs[0],
            xticklabels=target_names,
            yticklabels=target_names,
            annot_kws={"size": 13},
        )
        axs[0].set_title("Confusion Matrix", fontsize=12)
        axs[0].set_xlabel(r"Predicted Label $\hat{t}$", fontsize=11.5)
        axs[0].set_ylabel(r"True Target $t$", fontsize=11.5)

        colors = ["darkblue", "forestgreen", "crimson", "darkorange"]
        data_markers = ["s", "s", "s", "s"]
        fault_markers = ["x", "D", "o", "v"]

        d1, d2 = feat_idx
        for n, name in enumerate(target_names[: len(colors)]):
            t_idx = np.where(train_y_arr == name)[0]
            axs[1].scatter(
                train_X_arr[t_idx, d1],
                train_X_arr[t_idx, d2],
                color=colors[n],
                label=f"Train: {name}",
                marker=data_markers[n],
                s=48,
                alpha=0.6,
            )

        for n, name_true in enumerate(target_names[: len(colors)]):
            for k, name_pred in enumerate(target_names[: len(fault_markers)]):
                if name_true == name_pred:
                    continue
                fault_idx = np.where((pred_y_arr != test_y_arr) & (test_y_arr == name_true) & (pred_y_arr == name_pred))[0]
                if len(fault_idx) > 0:
                    axs[1].scatter(
                        test_X_arr[fault_idx, d1],
                        test_X_arr[fault_idx, d2],
                        marker=fault_markers[k % len(fault_markers)],
                        color=colors[n],
                        label=f"Fault: {name_true} $\\rightarrow$ {name_pred}",
                        s=72,
                        edgecolors="k",
                        lw=1.2,
                    )

        if feature_cols:
            l1 = feature_cols[d1] if d1 < len(feature_cols) else f"Feature {d1}"
            l2 = feature_cols[d2] if d2 < len(feature_cols) else f"Feature {d2}"
            axs[1].set_xlabel(l1, fontsize=11.5)
            axs[1].set_ylabel(l2, fontsize=11.5)
        else:
            axs[1].set_xlabel(f"Feature $x_{{{d1 + 1}}}$", fontsize=11.5)
            axs[1].set_ylabel(f"Feature $x_{{{d2 + 1}}}$", fontsize=11.5)

        axs[1].legend(loc="upper left", fontsize=8.5)
        axs[1].set_title("Empirical Classification & Test Faults", fontsize=12)
        axs[1].grid(True)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_decision_boundary(
        cls,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "Decision Boundary",
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """Render 2D classification decision surface and scatter observations."""
        standalone = False
        if ax is None:
            fig, ax_plot = cls.create_canvas(figsize=(7.5, 6.0))
            standalone = True
        else:
            ax_plot = ax
            fig = ax_plot.get_figure()

        X_arr = cls.to_numpy(X)
        y_arr = cls.to_numpy(y, ravel=True)

        xx, yy, mesh_pts = cls.create_2d_grid(X_arr, padding=0.5, resolution=250)
        Z = cls.eval_predictor(model, mesh_pts, as_tensor_like=X).reshape(xx.shape)

        ax_plot.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.coolwarm)

        unique_labels = np.unique(y_arr)
        palette = ["crimson", "dodgerblue", "forestgreen", "gold"]

        for idx, lbl in enumerate(unique_labels):
            mask = (y_arr == lbl)
            ax_plot.scatter(
                X_arr[mask, 0],
                X_arr[mask, 1],
                c=palette[idx % len(palette)],
                label=f"Class {lbl}",
                edgecolors="k",
                alpha=0.75,
                s=40,
            )

        ax_plot.set_title(title, fontsize=11.5)
        ax_plot.set_xlabel("$x_1$", fontsize=11.0)
        ax_plot.set_ylabel("$x_2$", fontsize=11.0)
        ax_plot.legend(loc="best", fontsize=9.5)
        ax_plot.grid(True)

        if standalone:
            return cls.finalize(fig, ax=ax_plot, show=show)
        return ax_plot

    @classmethod
    def plot_decision_boundaries_comparison(
        cls,
        models_dict: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        figsize: Tuple[float, float] = (14, 5.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Render 1xK side-by-side comparison panels of 2D decision boundaries."""
        n_models = len(models_dict)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]

        for ax, (title, model) in zip(axes, models_dict.items()):
            cls.plot_decision_boundary(model, X, y, ax=ax, title=title, show=False)

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_svm_panels(
        cls,
        panels: Dict[str, Tuple[Any, Any, Any, Any, Any, Any]],
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, Optional[plt.Axes]]]:
        """One panel per entry (X, t, xx, yy, Z, support_vectors): data, pre-computed decision values Z at -1, 0, +1, support vectors circled."""
        n = len(panels)
        n_cols = min(n, 2) if n != 3 else 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (4.6 * n_cols, 4.2 * n_rows), squeeze=False)
        for ax, (title, (X, t, xx, yy, Z, support_vectors)) in zip(axes.ravel(), panels.items()):
            X_arr, sv = _to_numpy(X), _to_numpy(support_vectors)
            ax.scatter(X_arr[:, 0], X_arr[:, 1], c=_to_numpy(t, ravel=True), cmap=cm_binary, edgecolors="k", s=36)
            ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["royalblue", "black", "royalblue"],
                       linestyles=["--", "-", "--"], linewidths=[1.6, 2.2, 1.6], alpha=0.8)
            if len(sv):
                ax.scatter(sv[:, 0], sv[:, 1], s=140, facecolors="none", edgecolors="firebrick", linewidths=1.6)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_title(title, fontsize=10.5)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_svm_separating_plane(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (5.5, 5.0),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Scatter linearly separable data with two hand-drawn candidate separating lines."""
        X_arr = _to_numpy(X)
        y_arr = _to_numpy(y)
        if xlim is None or ylim is None:
            xlim, ylim = cls.get_boundaries(X_arr, padding=1.0)

        fig, ax = cls.create_canvas(figsize=figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap=cm_binary, edgecolors="k", s=50)
        ax.plot(xlim, [ylim[1] - 2, ylim[0] + 2], "-c", lw=2, label="Candidate 1")
        ax.plot(xlim, [ylim[1] + 1, ylim[0] - 2], "-r", lw=2, label="Candidate 2")
        ax.set_title("Linearly Separable Data: Multiple Valid Lines", fontsize=11)
        ax.set_xlabel("$x_1$", fontsize=11)
        ax.set_ylabel("$x_2$", fontsize=11)
        ax.legend(loc="upper right", fontsize=9.5)
        ax.grid(True)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_feature_importances(
        cls,
        feature_names: Sequence[str],
        importances_dict: Dict[str, Sequence[float]],
        title: str = "Top Feature Importances (MDI)",
        figsize: Tuple[float, float] = (10, 4.5),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Render grouped horizontal bar charts comparing feature importances across ensemble models."""
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(feature_names))
        n_models = len(importances_dict)
        height = 0.8 / n_models
        colors = ["royalblue", "forestgreen", "darkorange", "firebrick"]

        for idx, (m_name, imp) in enumerate(importances_dict.items()):
            offset = (idx - n_models / 2 + 0.5) * height
            ax.barh(y_pos + offset, imp, height, label=m_name, color=colors[idx % len(colors)], alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11.5)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="lower right", fontsize=10.0)
        ax.grid(True)
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 4. Unsupervised Learning, PCA & Matrix Decompositions (07_unsupervised)
# ---------------------------------------------------------------------------

class AIMLUnsupervisedPlotter(BasePlotter):
    """Unsupervised visualizers: PCA scores, variance explained and dendrograms (biplot, directions, loadings: shared.PCAPlotter)."""

    @classmethod
    def plot_pca_scores(
        cls,
        X: np.ndarray,
        ort: np.ndarray,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 4.8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot projection of 2D data points onto a principal component orientation vector."""
        ort_arr = _to_numpy(ort).astype(np.float32).reshape(2, 1)
        X_arr = _to_numpy(X)
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
        if y_label:
            ax[0].set_ylabel(y_label, fontsize=11)

        for a in ax:
            if x_label:
                a.set_xlabel(x_label, fontsize=11)
            a.scatter(X_arr[:, 0], X_arr[:, 1], alpha=0.7, color="royalblue")
            l, r = a.get_xlim()
            x_line = np.linspace(l, r, 20)
            y_line = (ort_arr[1, 0] / ort_arr[0, 0]) * x_line if ort_arr[0, 0] != 0 else np.zeros_like(x_line)
            a.plot(x_line, y_line, c="darkorange", lw=2.2, label=r"PC Direction $\mathbf{u}_1$")
            a.grid(True)

        denom = cls.to_scalar(np.dot(ort_arr.T, ort_arr))
        proj_matrix = np.dot(ort_arr, ort_arr.T) / denom if denom > 0 else np.zeros((2, 2))
        for i in range(len(X_arr)):
            m = np.dot(proj_matrix, X_arr[i, :2].reshape(2, 1)).flatten()
            ax[1].plot([X_arr[i, 0], m[0]], [X_arr[i, 1], m[1]], c="forestgreen", alpha=0.4)
            ax[1].scatter(m[0], m[1], marker="x", c="k", s=30)

        ax[0].set_title(r"Principal Component Axis $\mathbf{u}_1$", fontsize=11.5)
        ax[1].set_title(r"Orthogonal Projection: $\mathbf{X}\mathbf{u}_1\mathbf{u}_1^\mathrm{T}$", fontsize=11.5)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_variance_explained(
        cls,
        ve: Sequence[float],
        figsize: Tuple[float, float] = (10, 4.2),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Plot individual and cumulative proportion of variance explained by principal components."""
        ve_arr = _to_numpy(ve).astype(float)
        total_var = float(ve_arr.sum())
        coeffs = ve_arr / total_var if total_var > 0 else ve_arr
        cumulative = np.cumsum(coeffs)
        k_range = list(range(1, len(ve_arr) + 1))

        fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True)
        ax[0].bar(k_range, ve_arr, color="royalblue")
        ax[0].set_ylabel("Variance Explained", fontsize=11.5)
        ax[0].set_xlabel("Principal Component", fontsize=11.5)
        ax[0].set_title("Individual Variance Explained", fontsize=12)
        ax[0].set_xticks(k_range)
        ax[0].grid(True)

        ax[1].bar(k_range, cumulative, color="forestgreen")
        ax[1].set_ylabel("Cumulative Variance Ratio", fontsize=11.5)
        ax[1].set_xlabel("Principal Component", fontsize=11.5)
        ax[1].set_ylim(0, 1.05)
        ax[1].set_title("Cumulative Proportion of Variance", fontsize=12)
        ax[1].set_xticks(k_range)
        ax[1].grid(True)

        return cls.finalize(fig, show=show)

    @classmethod
    def plot_dendrograms(
        cls,
        linkages: Dict[str, np.ndarray],
        labels: Optional[Sequence[str]] = None,
        cut_heights: Optional[Dict[str, float]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, Optional[plt.Axes]]]:
        """Dendrograms of pre-computed scipy linkage matrices, one panel each, with an optional horizontal cut line."""
        from scipy.cluster.hierarchy import dendrogram

        fig, axes = plt.subplots(len(linkages), 1, figsize=figsize or (12, 3.2 * len(linkages)), squeeze=False)
        for ax, (title, Z) in zip(axes[:, 0], linkages.items()):
            dendrogram(Z, labels=list(labels) if labels is not None else None, ax=ax, leaf_font_size=7, color_threshold=0)
            if cut_heights and title in cut_heights:
                ax.axhline(cut_heights[title], color="firebrick", linestyle="--", lw=1.2)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("fusion height")
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 5. Backward-Compatible Functional Facades & Aliases for simplified usage
# in the Jupyter Notebook environment
# ---------------------------------------------------------------------------

# Regression Facades
plot_residuals = AIMLRegressionPlotter.plot_residuals
plot_linear_classification_fit = AIMLRegressionPlotter.plot_linear_classification_fit
plot_logistic_response_curve = AIMLRegressionPlotter.plot_logistic_response_curve
plot_positive_vs_log = AIMLRegressionPlotter.plot_positive_vs_log
plot_curves = AIMLRegressionPlotter.plot_curves
plot_residual_panels = AIMLRegressionPlotter.plot_residual_panels
plot_confounding = AIMLRegressionPlotter.plot_confounding

# Classification Facades
get_boundaries = AIMLClassificationPlotter.get_boundaries
plot_knn_neighborhood_2d = AIMLClassificationPlotter.plot_knn_neighborhood_2d
plot_knn_classification_results = AIMLClassificationPlotter.plot_knn_classification_results
plot_class_counts = AIMLClassificationPlotter.plot_class_counts
plot_classes_2d = AIMLClassificationPlotter.plot_classes_2d
plot_decision_boundary = AIMLClassificationPlotter.plot_decision_boundary
plot_decision_boundaries_comparison = AIMLClassificationPlotter.plot_decision_boundaries_comparison
plot_svm_panels = AIMLClassificationPlotter.plot_svm_panels
plot_svm_separating_plane = AIMLClassificationPlotter.plot_svm_separating_plane
plot_feature_importances = AIMLClassificationPlotter.plot_feature_importances

# Unsupervised Facades
plot_pca_scores = AIMLUnsupervisedPlotter.plot_pca_scores
plot_variance_explained = AIMLUnsupervisedPlotter.plot_variance_explained
plot_dendrograms = AIMLUnsupervisedPlotter.plot_dendrograms

# Conversion & Helper Facades
to_numpy = BasePlotter.to_numpy
to_scalar = BasePlotter.to_scalar
ensure_2d = BasePlotter.ensure_2d
eval_predictor = BasePlotter.eval_predictor
create_2d_grid = BasePlotter.create_2d_grid

