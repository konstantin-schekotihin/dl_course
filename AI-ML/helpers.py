"""
Modular helper utilities for the AI and Machine Learning course (AI-ML).
Provides decoupled visualization, dataset resolution, and diagnostic routines:
1. Course Aesthetics & Dataset Path Resolution
2. Supervised Regression & Exploratory Diagnostics (02_regression)
3. Instance-Based Classification & k-NN Surfaces (01_classifiers-kNN)
4. Logistic Regression & Sigmoidal Projections (03_logistic_regression)
5. Decision Surfaces, Margin Geometry & Ensembles (04_NN, 05_ensembles, 06_SVM)
6. Unsupervised Learning, PCA & Matrix Decompositions (07_unsupervised)
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sklearn.metrics as metrics

# ---------------------------------------------------------------------------
# 1. Core Shared Foundation (Asset Resolution, Aesthetics, Data & Image Loaders)
# ---------------------------------------------------------------------------

try:
    from shared.helpers import (
        COURSE_RAW_URL,
        DATA_BASE_URL,
        annotate,
        compress_image_svd,
        get_data_path,
        get_file,
        get_url,
        load_cifar10,
        load_dataset,
        load_fashion_mnist,
        load_image,
        load_mnist,
        plot_binary,
        plot_correlation_matrix,
        plot_image,
        plot_pairgrid,
        plot_residuals,
        setup_theme,
    )
except ImportError:
    try:
        import sys
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if os.path.exists(os.path.join(_repo_root, "shared", "helpers.py")):
            if _repo_root not in sys.path:
                sys.path.insert(0, _repo_root)
            from shared.helpers import (
                COURSE_RAW_URL,
                DATA_BASE_URL,
                annotate,
                compress_image_svd,
                get_data_path,
                get_file,
                get_url,
                load_cifar10,
                load_dataset,
                load_fashion_mnist,
                load_image,
                load_mnist,
                plot_binary,
                plot_correlation_matrix,
                plot_image,
                plot_pairgrid,
                plot_residuals,
                setup_theme,
            )
        else:
            raise ImportError("Not in repository")
    except Exception:
        import urllib.request
        _shared_url = "https://raw.githubusercontent.com/konstantin-schekotihin/dl_course/master/shared/helpers.py"
        urllib.request.urlretrieve(_shared_url, "shared_helpers.py")
        from shared_helpers import (
            COURSE_RAW_URL,
            DATA_BASE_URL,
            annotate,
            compress_image_svd,
            get_data_path,
            get_file,
            get_url,
            load_cifar10,
            load_dataset,
            load_fashion_mnist,
            load_image,
            load_mnist,
            plot_binary,
            plot_correlation_matrix,
            plot_image,
            plot_pairgrid,
            plot_residuals,
            setup_theme,
        )

# Automatically configure unified course theme on import
setup_theme()


cm_binary: ListedColormap = ListedColormap(["green", "blue"])


def get_boundaries(
    X: Any, padding: float = 1.0
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute 2D coordinate plot boundaries with padding.

    Parameters
    ----------
    X : Any
        Feature matrix of shape (N, 2).
    padding : float, default=1.0
        Margin added to minimum and maximum values.

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        ((x_min, x_max), (y_min, y_max)) bounds.
    """
    X_arr = np.asarray(X)
    xlim = (float(np.min(X_arr[:, 0]) - padding), float(np.max(X_arr[:, 0]) + padding))
    ylim = (float(np.min(X_arr[:, 1]) - padding), float(np.max(X_arr[:, 1]) + padding))
    return xlim, ylim


# ---------------------------------------------------------------------------
# 2. Supervised Regression & Exploratory Diagnostics (02_regression)
# ---------------------------------------------------------------------------

def annotate(x: Sequence[float], y: Sequence[float], **kws: Any) -> None:
    """
    Annotate Seaborn PairGrid subplots with Pearson correlation coefficient.

    Parameters
    ----------
    x : Sequence[float]
        First numerical variable.
    y : Sequence[float]
        Second numerical variable.
    **kws : Any
        Additional keyword arguments from PairGrid mapping.
    """
    r, p = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(f"$r = {r:.2f}, p = {p:.3f}$", xy=(0.1, 0.9), xycoords=ax.transAxes, fontsize=11)


def plot_residuals(
    f: Union[Callable[[Any], Any], Any],
    pred: Any,
    resp: Any,
    show: bool = False,
    xlabel: str = "TV Budget ($x$)",
    ylabel: str = "Sales ($t$)",
    title: str = "Linear Regression Residuals",
    figsize: Tuple[float, float] = (8, 5.5),
) -> None:
    """
    Plot regression observations, fitted model curve, and vertical residual drop-lines.

    Parameters
    ----------
    f : Callable or array-like
        Fitted predictor function or array of estimated target values.
    pred : array-like
        Observed feature input values.
    resp : array-like
        Observed true target values.
    show : bool, default=False
        If True, prints summary regression metrics (TSS, RSS, ESS, MSE, MAE).
    xlabel : str, default="TV Budget ($x$)"
        Horizontal axis label.
    ylabel : str, default="Sales ($t$)"
        Vertical axis label.
    title : str, default="Linear Regression Residuals"
        Figure title.
    figsize : Tuple[float, float], default=(8, 5.5)
        Matplotlib figure dimensions.
    """
    pred_arr = np.asarray(pred).ravel()
    resp_arr = np.asarray(resp).ravel()

    if callable(f):
        try:
            est_arr = np.asarray(f(pred_arr.reshape(-1, 1))).ravel()
        except Exception:
            est_arr = np.asarray(f(pred_arr)).ravel()
    else:
        est_arr = np.asarray(f).ravel()

    if show:
        rss = float(np.sum((resp_arr - est_arr) ** 2))
        tss = float(np.sum((resp_arr - np.mean(resp_arr)) ** 2))
        ess = tss - rss
        mse = rss / len(resp_arr)
        mae = float(np.mean(np.abs(resp_arr - est_arr)))
        display(Markdown(rf"$\mathrm{{TSS}} = {tss:.3f}$ (total sum of squares)"))
        display(Markdown(rf"$\mathrm{{RSS}} = {rss:.3f}$ (residual sum of squares)"))
        display(Markdown(rf"$\mathrm{{ESS}} = \mathrm{{TSS}} - \mathrm{{RSS}} = {ess:.3f}$ (explained sum of squares)"))
        display(Markdown(rf"$\mathrm{{MSE}} = {mse:.3f}$ (mean squared error)"))
        display(Markdown(rf"$\mathrm{{MAE}} = {mae:.3f}$ (mean absolute error)"))

    x_min, x_max = float(pred_arr.min()), float(pred_arr.max())
    span = max(x_max - x_min, 1.0)
    l = np.linspace(x_min - 0.05 * span, x_max + 0.05 * span, num=500)

    plt.figure(figsize=figsize)
    plt.vlines(pred_arr, resp_arr, est_arr, colors="royalblue", linestyles="dashed", alpha=0.5, label="Residuals")
    plt.scatter(pred_arr, resp_arr, color="royalblue", alpha=0.6, s=40, label="Observations")
    plt.scatter(pred_arr, est_arr, color="firebrick", s=30, label="Predictions")

    if callable(f):
        try:
            l_pred = np.asarray(f(l.reshape(-1, 1))).ravel()
        except Exception:
            l_pred = np.asarray(f(l)).ravel()
        plt.plot(l, l_pred, "firebrick", lw=2.2, label=r"Fitted $y(x, \mathbf{w})$")

    plt.title(title, fontsize=12.5)
    plt.xlabel(xlabel, fontsize=11.5)
    plt.ylabel(ylabel, fontsize=11.5)
    plt.legend(loc="upper left", fontsize=10.0)
    plt.tight_layout()
    plt.show()


def plot_pairgrid(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    annotate_pearson: bool = True,
    figsize: Tuple[float, float] = (7.5, 6.5),
) -> sns.PairGrid:
    """
    Render styled PairGrid with upper scatter + Pearson r, diagonal histogram, and lower KDE contours.

    Parameters
    ----------
    df : pd.DataFrame
        Input data table.
    cols : Optional[Sequence[str]], default=None
        Subset of dataframe columns to include.
    annotate_pearson : bool, default=True
        Whether to overlay Pearson r annotations on upper scatter subplots.
    figsize : Tuple[float, float], default=(7.5, 6.5)
        Target figure dimensions.

    Returns
    -------
    sns.PairGrid
        Configured Seaborn PairGrid object.
    """
    data = df[list(cols)] if cols is not None else df
    g = sns.PairGrid(data)
    g.fig.set_size_inches(figsize[0], figsize[1])
    g.map_upper(plt.scatter, s=12, alpha=0.6, color="royalblue")
    if annotate_pearson:
        g.map_upper(annotate)
    g.map_diag(sns.histplot, kde=False, color="cornflowerblue")
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    plt.tight_layout()
    plt.show()
    return g


def plot_correlation_matrix(
    corr_matrix: Any,
    labels: Optional[Sequence[str]] = None,
    title: str = "Correlation Matrix",
    figsize: Tuple[float, float] = (5.5, 4.8),
) -> None:
    """
    Render styled correlation matrix heatmap.

    Parameters
    ----------
    corr_matrix : Any
        Square correlation matrix (NumPy array or pandas DataFrame).
    labels : Optional[Sequence[str]], default=None
        Axis tick labels.
    title : str, default="Correlation Matrix"
        Plot title.
    figsize : Tuple[float, float], default=(5.5, 4.8)
        Matplotlib figure dimensions.
    """
    plt.figure(figsize=figsize)
    df_corr = pd.DataFrame(corr_matrix) if not isinstance(corr_matrix, pd.DataFrame) else corr_matrix
    if labels is not None:
        df_corr.columns = list(labels)
        df_corr.index = list(labels)

    sns.heatmap(
        df_corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        vmin=-1.0,
        vmax=1.0,
        cbar=True,
        annot_kws={"size": 11},
    )
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Instance-Based Classification & k-NN Surfaces (01_classifiers-kNN)
# ---------------------------------------------------------------------------

def plot_knn_neighborhood_2d(
    X: Any,
    y: Any,
    query_pt: Tuple[float, float] = (5.5, 4.7),
    radius: float = 0.3,
    feature_names: Tuple[str, str] = ("Sepal Length ($x_1$)", "Petal Length ($x_2$)"),
    show_neighborhood: bool = False,
    figsize: Tuple[float, float] = (7.5, 6.5),
) -> None:
    """
    Render 2D botanical scatter with decision query point and radius circle.

    Parameters
    ----------
    X : Any
        2D feature coordinates array or DataFrame.
    y : Any
        Target class labels.
    query_pt : Tuple[float, float], default=(5.5, 4.7)
        Coordinates of query instance.
    radius : float, default=0.3
        Metric ball radius for neighborhood intuition.
    feature_names : Tuple[str, str], default=("Sepal Length ($x_1$)", "Petal Length ($x_2$)")
        Labels for horizontal and vertical axes.
    show_neighborhood : bool, default=False
        Whether to highlight the query point and radius disk.
    figsize : Tuple[float, float], default=(7.5, 6.5)
        Figure size.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    unique_classes = np.unique(y_arr)
    palette = sns.color_palette("deep", n_colors=len(unique_classes))

    for idx, cls in enumerate(unique_classes):
        mask = (y_arr == cls)
        ax.scatter(
            X_arr[mask, 0],
            X_arr[mask, 1],
            color=palette[idx],
            label=f"Class: {cls}",
            edgecolors="w",
            s=50,
            alpha=0.85,
        )

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_title("Instance-Based Learning: Nearest Neighbor Radius", fontsize=12.5)

    if show_neighborhood:
        ax.scatter([query_pt[0]], [query_pt[1]], color="magenta", s=100, zorder=5, label="Query Point $\\mathbf{x}_q$")
        circle = plt.Circle(query_pt, radius=radius, color="magenta", alpha=0.25, zorder=4)
        ax.add_artist(circle)

    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_knn_intuition(iris: Any, show: bool = False) -> None:
    """
    Backward-compatible adapter for Iris kNN intuition illustration.

    Parameters
    ----------
    iris : Any
        Iris dataset DataFrame with sepal_length, petal_length, and species columns.
    show : bool, default=False
        Whether to overlay the query circle.
    """
    X = np.c_[iris.sepal_length.values, iris.petal_length.values]
    y = iris.species.values
    plot_knn_neighborhood_2d(
        X,
        y,
        query_pt=(5.5, 4.7),
        radius=0.3,
        feature_names=("Sepal Length ($x_1$)", "Petal Length ($x_2$)"),
        show_neighborhood=show,
    )


def plot_knn_classification_results(
    train_X: Any,
    train_y: Any,
    test_X: Any,
    test_y: Any,
    pred_y: Any,
    target_names: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    feat_idx: Tuple[int, int] = (3, 0),
    figsize: Tuple[float, float] = (14, 5.5),
) -> None:
    """
    Dual-subplot evaluation for kNN: Left = Confusion Matrix; Right = 2D Scatter with Misclassification Markers.

    Parameters
    ----------
    train_X : Any
        Training feature matrix.
    train_y : Any
        Training class labels.
    test_X : Any
        Testing feature matrix.
    test_y : Any
        Ground-truth testing class labels.
    pred_y : Any
        Predicted testing class labels.
    target_names : Optional[Sequence[str]], default=None
        Ordered list of class names.
    feature_cols : Optional[Sequence[str]], default=None
        Names of all input feature dimensions.
    feat_idx : Tuple[int, int], default=(3, 0)
        Indices of the two feature columns displayed in 2D scatter.
    figsize : Tuple[float, float], default=(14, 5.5)
        Figure dimensions.
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    train_y_arr = np.asarray(train_y)
    test_y_arr = np.asarray(test_y)
    pred_y_arr = np.asarray(pred_y)
    train_X_arr = np.asarray(train_X)
    test_X_arr = np.asarray(test_X)

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
    plt.tight_layout()
    plt.show()


def plot_knn_results(
    train_X: Any,
    train_y: Any,
    test_X: Any,
    test_y: Any,
    pred_y: Any,
    target_names: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> None:
    """Backward-compatible alias for plot_knn_classification_results."""
    plot_knn_classification_results(
        train_X,
        train_y,
        test_X,
        test_y,
        pred_y,
        target_names=target_names,
        feature_cols=feature_cols,
        feat_idx=(3, 0),
    )


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (6.5, 5.5),
) -> None:
    """
    Render styled confusion matrix heatmap from precomputed true and predicted labels.

    Parameters
    ----------
    y_true : Any
        Ground-truth labels.
    y_pred : Any
        Predicted labels.
    labels : Optional[Sequence[str]], default=None
        Display class names or label subset.
    title : str, default="Confusion Matrix"
        Plot title.
    figsize : Tuple[float, float], default=(6.5, 5.5)
        Figure dimensions.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    unique_vals = np.unique(np.concatenate([y_t, y_p]))

    if labels is not None and len(labels) == len(unique_vals) and not np.array_equal(unique_vals, labels):
        cm = metrics.confusion_matrix(y_t, y_p, labels=unique_vals)
        display_labels = list(labels)
    elif labels is not None:
        try:
            cm = metrics.confusion_matrix(y_t, y_p, labels=labels)
            display_labels = list(labels)
        except Exception:
            cm = metrics.confusion_matrix(y_t, y_p)
            display_labels = list(labels)
    else:
        cm = metrics.confusion_matrix(y_t, y_p)
        display_labels = list(unique_vals)

    plt.figure(figsize=figsize)
    df_cm = pd.DataFrame(cm, columns=display_labels, index=display_labels)
    df_cm.index.name = "True Label $t$"
    df_cm.columns.name = r"Predicted $\hat{t}$"
    sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="d", annot_kws={"size": 14})
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Logistic Regression & Sigmoidal Projections (03_logistic_regression)
# ---------------------------------------------------------------------------

def plot_linear_classification_fit(
    X: Any,
    y: Any,
    predict_fn: Callable[[Any], Any],
    xlabel: str = r"Standardized Balance ($x$)",
    ylabel: str = r"Class Target ($t$)",
    title: str = "Linear Regression on Binary Classification",
    figsize: Tuple[float, float] = (8, 4.8),
) -> None:
    """
    Plot scatter observations and unbounded linear regression predictor.

    Parameters
    ----------
    X : Any
        Input feature vector or matrix.
    y : Any
        Binary targets (0 and 1).
    predict_fn : Callable
        Fitted linear prediction model callable.
    xlabel : str, default="Standardized Balance ($x$)"
        Horizontal axis label.
    ylabel : str, default="Class Target ($t$)"
        Vertical axis label.
    title : str
        Figure title.
    figsize : Tuple[float, float], default=(8, 4.8)
        Figure dimensions.
    """
    X_arr = np.asarray(X).ravel()
    y_arr = np.asarray(y).ravel()
    span = max(float(X_arr.max() - X_arr.min()), 1.0)
    l = np.linspace(X_arr.min() - 0.1 * span, X_arr.max() + 0.1 * span, num=200).reshape(-1, 1)

    plt.figure(figsize=figsize)
    plt.scatter(X_arr, y_arr, alpha=0.5, color="royalblue", label="Observations ($x_n, t_n$)")
    plt.plot(l, predict_fn(l), "firebrick", lw=2.2, label=r"Linear fit: $y(x, \mathbf{w}) = \mathbf{w}^\mathrm{T}\mathbf{x}$")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Threshold $y=0.5$")
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=11.5)
    plt.ylabel(ylabel, fontsize=11.5)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.show()


def plot_linear_fit(X: Any, y: Any, predict_fn: Callable[[Any], Any]) -> None:
    """Backward-compatible alias for plot_linear_classification_fit."""
    plot_linear_classification_fit(X, y, predict_fn)


def plot_logistic_response_curve(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    w_0: Optional[float] = None,
    w_1: Optional[float] = None,
    xlabel: str = r"Standardized Balance ($x$)",
    ylabel: str = r"Predicted Probability $p(t=1 \mid x)$",
    title: str = "Logistic Sigmoid Response Curve",
    figsize: Tuple[float, float] = (8, 4.8),
) -> None:
    """
    Plot logistic sigmoid response curve over feature range with decision threshold.

    Parameters
    ----------
    x_vals : Sequence[float]
        Feature values across grid.
    y_vals : Sequence[float]
        Evaluated sigmoid probabilities.
    w_0 : Optional[float], default=None
        Bias parameter for equation badge.
    w_1 : Optional[float], default=None
        Weight parameter for equation badge.
    xlabel : str
        Horizontal axis label.
    ylabel : str
        Vertical axis label.
    title : str
        Figure title.
    figsize : Tuple[float, float]
        Figure dimensions.
    """
    plt.figure(figsize=figsize)
    if w_0 is not None and w_1 is not None:
        eq_label = rf"$\sigma({w_0:.1f} + {w_1:.1f}x)$"
    else:
        eq_label = r"$\sigma(a) = \frac{1}{1 + e^{-a}}$"

    plt.plot(x_vals, y_vals, "royalblue", lw=2.4, label=eq_label)
    plt.axhline(0.5, color="firebrick", linestyle="--", label="Decision threshold $p = 0.5$")
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=11.5)
    plt.ylabel(ylabel, fontsize=11.5)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper left", fontsize=10.0)
    plt.tight_layout()
    plt.show()


def plot_logistic_curve(x_vals: Sequence[float], y_vals: Sequence[float], w_0: float, w_1: float) -> None:
    """Backward-compatible alias for plot_logistic_response_curve."""
    plot_logistic_response_curve(x_vals, y_vals, w_0=w_0, w_1=w_1)


def plot_positive_vs_log(
    l: Any,
    f_vals: Any,
    log_vals: Optional[Any],
    is_positive: bool = True,
    figsize: Tuple[float, float] = (8, 4.8),
) -> None:
    """
    Plot positive surrogate function f and its monotonic log transform.

    Parameters
    ----------
    l : Any
        Feature input values.
    f_vals : Any
        Surrogate function values.
    log_vals : Optional[Any]
        Logarithm of function values.
    is_positive : bool, default=True
        Whether the base function is strictly positive on domain.
    figsize : Tuple[float, float], default=(8, 4.8)
        Figure dimensions.
    """
    plt.figure(figsize=figsize)
    plt.plot(l, f_vals, "royalblue", lw=2.2, label=r"Likelihood surrogate $f(x)$")
    if not is_positive:
        plt.title(r"Warning: The function is not strictly positive ($f(x) \le 0$)", fontsize=12)
    elif log_vals is not None:
        plt.plot(l, log_vals, "firebrick", lw=2.0, linestyle="--", label=r"Log-likelihood $\ln f(x)$")
        plt.title(r"Monotonicity of the Logarithmic Transformation: $\arg\max f(x) = \arg\max \ln f(x)$", fontsize=11.5)
    plt.xlabel("$x$", fontsize=11.5)
    plt.ylabel("Value", fontsize=11.5)
    plt.legend(loc="best", fontsize=10.0)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Decision Surfaces, Margins & Ensembles (04_NN, 05_ensembles, 06_SVM)
# ---------------------------------------------------------------------------

def plot_binary(
    predictor: Callable[[Any], Any],
    X: Any,
    y: Any,
    figsize: Tuple[float, float] = (7.5, 6.5),
) -> None:
    """
    Plot decision boundaries for 2D binary classification (Perceptron / Adaline).

    Parameters
    ----------
    predictor : Callable
        Decision rule callable mapping 2D inputs to predictions.
    X : Any
        Input coordinates of shape (N, 2).
    y : Any
        Binary targets {-1, +1} or {0, 1}.
    figsize : Tuple[float, float], default=(7.5, 6.5)
        Figure dimensions.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    xlim, ylim = get_boundaries(X_arr, padding=0.5)
    xm, ym = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 200),
        np.linspace(ylim[0], ylim[1], 200),
    )
    mesh_in = np.c_[xm.ravel(), ym.ravel()]
    try:
        import torch
        if isinstance(X, torch.Tensor):
            p = predictor(torch.tensor(mesh_in, dtype=X.dtype))
            if hasattr(p, "detach"):
                p = p.detach().cpu().numpy()
        else:
            p = predictor(mesh_in)
    except Exception:
        p = predictor(mesh_in)

    p_arr = np.asarray(p).reshape(xm.shape)

    plt.figure(figsize=figsize)
    plt.contourf(xm, ym, p_arr, cmap="coolwarm", alpha=0.3, levels=np.linspace(p_arr.min(), p_arr.max(), 30))
    plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap="coolwarm", edgecolors="k", s=80, linewidths=1.2)
    plt.title(r"2D Binary Decision Surface: $y(\mathbf{x}) = f(\mathbf{w}^\mathrm{T}\mathbf{x} + w_0)$", fontsize=12)
    plt.xlabel("$x_1$", fontsize=11.5)
    plt.ylabel("$x_2$", fontsize=11.5)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    model: Any,
    X: Any,
    y: Any,
    ax: Optional[Any] = None,
    title: str = "Decision Boundary",
) -> None:
    """
    Render 2D classification decision surface and scatter observations.
    Supports both standalone rendering (ax=None) and subplot integration.

    Parameters
    ----------
    model : Any
        Trained model or pipeline with predict / predict_proba or callable.
    X : Any
        Feature matrix of shape (N, 2).
    y : Any
        Class labels.
    ax : Optional[matplotlib.axes.Axes], default=None
        Subplot axis. If None, creates a new figure.
    title : str, default="Decision Boundary"
        Axis title.
    """
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        standalone = True

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    xlim, ylim = get_boundaries(X_arr, padding=0.5)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 250), np.linspace(ylim[0], ylim[1], 250))
    mesh_pts = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(model, "predict"):
        Z = model.predict(mesh_pts).reshape(xx.shape)
    elif hasattr(model, "predict_proba"):
        Z = model.predict_proba(mesh_pts)[:, 1].reshape(xx.shape)
    else:
        Z = model(mesh_pts).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.coolwarm)
    unique_labels = np.unique(y_arr)
    palette = ["crimson", "dodgerblue", "forestgreen", "gold"]

    for idx, lbl in enumerate(unique_labels):
        mask = (y_arr == lbl)
        ax.scatter(
            X_arr[mask, 0],
            X_arr[mask, 1],
            c=palette[idx % len(palette)],
            label=f"Class {lbl}",
            edgecolors="k",
            alpha=0.75,
            s=40,
        )

    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("$x_1$", fontsize=11.0)
    ax.set_ylabel("$x_2$", fontsize=11.0)
    ax.legend(loc="best", fontsize=9.5)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_decision_boundaries_comparison(
    models_dict: Dict[str, Any],
    X: Any,
    y: Any,
    figsize: Tuple[float, float] = (14, 5.0),
) -> None:
    """
    Render 1xK side-by-side comparison panels of 2D decision boundaries.

    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary mapping panel titles to trained models.
    X : Any
        Feature matrix (N, 2).
    y : Any
        Class labels.
    figsize : Tuple[float, float], default=(14, 5.0)
        Figure dimensions.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (title, model) in zip(axes, models_dict.items()):
        plot_decision_boundary(model, X, y, ax=ax, title=title)

    plt.tight_layout()
    plt.show()


def plot_svm_margins(
    model: Any,
    ax: Optional[Any] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot SVM decision boundary, margin contours (Z = -1, 0, +1), and support vectors.

    Parameters
    ----------
    model : Any
        Trained SVM model with decision_function and optional support_vectors_.
    ax : Optional[Any], default=None
        Matplotlib Axes or None. Handles ax=plt compatibility.
    xlim : Optional[Tuple[float, float]], default=None
        Horizontal limits.
    ylim : Optional[Tuple[float, float]], default=None
        Vertical limits.
    """
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        standalone = True
    elif ax is plt:
        ax = plt.gca()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 80)
    yy = np.linspace(ylim[0], ylim[1], 80)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(
        XX,
        YY,
        Z,
        colors=["royalblue", "black", "royalblue"],
        levels=[-1, 0, 1],
        alpha=0.6,
        linestyles=["--", "-", "--"],
        linewidths=[1.8, 2.2, 1.8],
    )
    if hasattr(model, "support_vectors_") and len(model.support_vectors_) > 0:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors="none",
            edgecolors="firebrick",
            s=120,
            linewidths=1.8,
            label="Support Vectors",
            zorder=6,
        )
        ax.legend(loc="lower right", fontsize=9.5)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_svm_separating_plane(
    X: Any,
    y: Any,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_margin: bool = False,
    figsize: Tuple[float, float] = (5.5, 5.0),
) -> None:
    """
    Render separating hyperplane intuition and shaded half-spaces.

    Parameters
    ----------
    X : Any
        2D data coordinates.
    y : Any
        Binary class targets.
    xlim : Optional[Tuple[float, float]], default=None
        X limits.
    ylim : Optional[Tuple[float, float]], default=None
        Y limits.
    show_margin : bool, default=False
        Whether to show optimal separating plane vs candidate planes.
    figsize : Tuple[float, float], default=(5.5, 5.0)
        Figure size.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if xlim is None or ylim is None:
        xlim, ylim = get_boundaries(X_arr, padding=1.0)

    plt.figure(figsize=figsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap=cm_binary, edgecolors="k", s=50)

    if not show_margin:
        plt.plot(xlim, [ylim[1] - 2, ylim[0] + 2], "-c", lw=2, label="Candidate 1")
        plt.plot(xlim, [ylim[1] + 1, ylim[0] - 2], "-r", lw=2, label="Candidate 2")
        plt.title("Linearly Separable Data: Multiple Valid Planes", fontsize=11)
    else:
        plt.plot(xlim, [ylim[1], ylim[0]], "-k", lw=2.4, label="Separating Plane")
        plt.fill_between(xlim, [ylim[1], ylim[0]], ylim[0], alpha=0.15, color="blue")
        plt.title("Maximum Margin Separating Hyperplane", fontsize=11)

    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    plt.show()


def plot_feature_importances(
    feature_names: Sequence[str],
    importances_dict: Dict[str, Sequence[float]],
    title: str = "Top Feature Importances (MDI)",
    figsize: Tuple[float, float] = (10, 4.5),
) -> None:
    """
    Render grouped horizontal bar charts comparing feature importances across ensemble models.

    Parameters
    ----------
    feature_names : Sequence[str]
        Ordered feature labels.
    importances_dict : Dict[str, Sequence[float]]
        Dictionary mapping model names to importance scores.
    title : str, default="Top Feature Importances (MDI)"
        Figure title.
    figsize : Tuple[float, float], default=(10, 4.5)
        Figure dimensions.
    """
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
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Unsupervised Learning, PCA & Matrix Decompositions (07_unsupervised)
# ---------------------------------------------------------------------------

def plot_pca_scores(
    X: Any,
    ort: Any,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4.8),
) -> None:
    """
    Plot projection of 2D data points onto a principal component orientation vector.

    Parameters
    ----------
    X : Any
        Feature matrix (N, 2).
    ort : Any
        Unit orientation vector (2, 1).
    x_label : Optional[str]
        Horizontal axis label.
    y_label : Optional[str]
        Vertical axis label.
    figsize : Tuple[float, float]
        Figure size.
    """
    ort_arr = np.asarray(ort, dtype=np.float32).reshape(2, 1)
    X_arr = np.asarray(X)
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

    denom = float(np.dot(ort_arr.T, ort_arr).item())
    proj_matrix = np.dot(ort_arr, ort_arr.T) / denom if denom > 0 else np.zeros((2, 2))
    for i in range(len(X_arr)):
        m = np.dot(proj_matrix, X_arr[i, :2].reshape(2, 1)).flatten()
        ax[1].plot([X_arr[i, 0], m[0]], [X_arr[i, 1], m[1]], c="forestgreen", alpha=0.4)
        ax[1].scatter(m[0], m[1], marker="x", c="k", s=30)

    ax[0].set_title(r"Principal Component Axis $\mathbf{u}_1$", fontsize=11.5)
    ax[1].set_title(r"Orthogonal Projection: $\mathbf{X}\mathbf{u}_1\mathbf{u}_1^\mathrm{T}$", fontsize=11.5)
    fig.tight_layout()
    plt.show()


def plot_variance_explained(
    ve: Sequence[float],
    figsize: Tuple[float, float] = (10, 4.2),
) -> None:
    """
    Plot individual and cumulative proportion of variance explained by principal components.

    Parameters
    ----------
    ve : Sequence[float]
        Array of variance explained values per component.
    figsize : Tuple[float, float], default=(10, 4.2)
        Figure dimensions.
    """
    ve_arr = np.asarray(ve, dtype=float)
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

    ax[1].bar(k_range, cumulative, color="forestgreen")
    ax[1].set_ylabel("Cumulative Variance Ratio", fontsize=11.5)
    ax[1].set_xlabel("Principal Component", fontsize=11.5)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Cumulative Proportion of Variance", fontsize=12)
    ax[1].set_xticks(k_range)

    fig.tight_layout()
    plt.show()


def plot_pca_biplot(
    z1: int,
    z2: int,
    sc: Any,
    comps: Any,
    obs: Sequence[Any],
    features: Sequence[str],
    colors: Sequence[str],
    figsize: Tuple[float, float] = (9.5, 8.5),
) -> None:
    """
    Render PCA biplot with observation scores and feature loading vectors.

    Parameters
    ----------
    z1 : int
        First principal component index (0-indexed).
    z2 : int
        Second principal component index (0-indexed).
    sc : Any
        Observation score matrix (N, K).
    comps : Any
        Component loading matrix (K, D).
    obs : Sequence[Any]
        Observation labels (e.g. US state names).
    features : Sequence[str]
        Original feature names.
    colors : Sequence[str]
        Color palette for loading arrows.
    figsize : Tuple[float, float], default=(9.5, 8.5)
        Figure size.
    """
    x = np.asarray(sc)[:, z1]
    y = np.asarray(sc)[:, z2]
    comps_arr = np.asarray(comps)

    fig = plt.figure(figsize=figsize)
    plt.xlabel(f"$z_{{{z1 + 1}}}$", fontsize=12.5)
    plt.ylabel(f"$z_{{{z2 + 1}}}$", fontsize=12.5)

    sx = float((x.max() - x.min()) / 2)
    sy = float((y.max() - y.min()) / 2)

    plt.scatter(x, y, alpha=0.5, color="royalblue")
    for i in range(len(obs)):
        plt.text(x[i], y[i], str(obs[i]), ha="center", fontsize=10, alpha=0.75)

    vec = comps_arr[[z1, z2], :].T
    for i in range(len(vec)):
        c = colors[i % len(colors)]
        plt.arrow(
            0,
            0,
            vec[i, 0] * sx,
            vec[i, 1] * sy,
            ec=c,
            head_width=0.08,
            head_length=0.08,
            fc=c,
            lw=1.6,
        )
        plt.text(
            vec[i, 0] * sx * 1.15,
            vec[i, 1] * sy * 1.15,
            features[i],
            color=c,
            fontsize=11.5,
            fontweight="bold",
        )

    plt.title(f"Principal Component Biplot: $z_{{{z1 + 1}}}$ vs $z_{{{z2 + 1}}}$", fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loading_heatmap(
    loadings: Any,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    title: str = "Principal Component Loadings",
    figsize: Tuple[float, float] = (8, 4.2),
) -> None:
    """
    Render styled diverging heatmap for principal component loading matrices.

    Parameters
    ----------
    loadings : Any
        Loading matrix of shape (n_components, n_features).
    x_labels : Optional[Sequence[str]], default=None
        Feature column names.
    y_labels : Optional[Sequence[str]], default=None
        Principal component labels (e.g. ["$z_1$", "$z_2$"]).
    title : str, default="Principal Component Loadings"
        Heatmap title.
    figsize : Tuple[float, float], default=(8, 4.2)
        Figure dimensions.
    """
    loadings_arr = np.asarray(loadings)
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(10, 240, as_cmap=True)

    if y_labels is None:
        y_labels = [f"$z_{i+1}$" for i in range(len(loadings_arr))]

    sns.heatmap(
        loadings_arr,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        xticklabels=list(x_labels) if x_labels is not None else True,
        yticklabels=list(y_labels),
        cbar=True,
    )
    plt.title(title, fontsize=12.5)
    plt.tight_layout()
    plt.show()




# Backward-compatible function aliases
plot_var_exp = plot_variance_explained
biplot = plot_pca_biplot
compress = compress_image_svd

