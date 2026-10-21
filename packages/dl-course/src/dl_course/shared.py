"""
Canonical Shared Course Helpers & Infrastructure (dl_course / shared).

Provides unified asset resolution, aesthetics, image utilities, and diagnostic visualizers.
Inherited and re-exported by course-specific helper modules across ML-DL and AI-ML.

Modules:
1. Course Aesthetics & Asset Resolution (get_file, get_url, get_data_path, setup_theme, print_math, format_number, format_vector)
2. Base & Specialized Plotter Classes (BasePlotter, RegressionPlotter, BoundaryPlotter, DiagnosticPlotter, ClassificationPlotter, ImagePlotter, ActivationPlotter, PCAPlotter)
3. Image Loading, Processing & Display (load_image, plot_image, compress_image_svd)
4. Functional Facades (plot_residuals, plot_binary, plot_correlation_matrix, plot_pairgrid, annotate)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

# torch, PIL and scipy are imported only inside the helpers that need them, so that
# importing the course helpers stays fast (e.g. AI-ML notebooks that never use torch).
if TYPE_CHECKING:
    import torch
    from PIL import Image


# ---------------------------------------------------------------------------
# 1. Course Aesthetics & Asset Resolution
# ---------------------------------------------------------------------------

COURSE_RAW_URL: str = os.environ.get(
    "COURSE_RAW_URL",
    "https://raw.githubusercontent.com/konstantin-schekotihin/dl_course/master/shared",
)

DATA_BASE_URL: str = os.environ.get(
    "COURSE_DATA_URL",
    f"{COURSE_RAW_URL}/data",
)


def print_math(text: str) -> None:
    """Display formatted LaTeX Markdown output in Jupyter environments."""
    display(Markdown(text))


def format_number(value: Any, digits: int = 4, scientific: bool = False) -> str:
    """Number for display inside a LaTeX formula, with infinity as a symbol; scientific=True writes a \\cdot 10^b for values below 0.01 or from 1000 on."""
    value = float(value)
    if np.isinf(value):
        return r"-\infty" if value < 0 else r"\infty"
    if scientific:
        if value == 0.0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        if -3 < exponent < 3:  # 0.056, 0.56, 5.6, 560: plain digits read better than a power of ten
            return np.format_float_positional(float(f"{value:.{digits + 1}g}"), trim="-")
        mantissa = round(value / 10.0**exponent, digits)
        if abs(mantissa) == 10.0:  # rounding carried into the next power
            mantissa, exponent = mantissa / 10.0, exponent + 1
        if exponent == 0:
            return np.format_float_positional(mantissa, trim="-")
        power = f"10^{{{exponent}}}"
        if mantissa == 1.0:
            return power
        return np.format_float_positional(mantissa, trim="-") + r" \cdot " + power
    return f"{round(value, digits) + 0.0:.{digits}f}"  # + 0.0 turns -0.0 into 0.0


def format_vector(values: Any, digits: int = 4, transpose: bool = True) -> str:
    """Vector for display inside a LaTeX formula: (v_1, ..., v_n)^T, or (v_1, ..., v_n) with transpose=False.

    Entries are rounded to `digits` decimals without trailing zeros (3.0 -> 3, 0.50 -> 0.5).
    """
    def entry(v: float) -> str:
        if np.isinf(v):
            return format_number(v)
        if np.isnan(v):
            return r"\mathrm{NaN}"
        return np.format_float_positional(round(v, digits) + 0.0, trim="-")  # + 0.0 turns -0.0 into 0.0

    text = "(" + ", ".join(entry(float(v)) for v in np.asarray(values, dtype=float).ravel()) + ")"
    return text + r"^\mathrm{T}" if transpose else text


def get_url(filename: str, category: Optional[str] = None) -> str:
    """
    Construct canonical remote repository URL for a course asset.

    Parameters
    ----------
    filename : str
        Basename of the file (e.g. 'puppy.jpeg', 'Advertising.csv').
    category : Optional[str], default=None
        Asset directory under 'shared/' ('images' or 'data').
        If None, automatically inferred from file extension.

    Returns
    -------
    str
        Canonical HTTPS URL to the raw asset on GitHub.
    """
    basename = os.path.basename(filename)
    category = category or _category(basename)
    return f"{COURSE_RAW_URL}/{category}/{basename}"


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".tif", ".tiff"}


def _category(filename: str) -> str:
    """Infer the 'shared/' asset directory ('images' or 'data') from a file extension."""
    return "images" if os.path.splitext(filename)[1].lower() in _IMAGE_EXTENSIONS else "data"


def _find_shared_dir(category: str) -> Optional[str]:
    """
    Locate the repository's 'shared/<category>' directory by searching the working
    directory and its parents (notebooks run from course subfolders such as 'ML-DL/04_...').
    """
    directory = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(directory, "shared", category)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def get_file(filename: str = "", category: Optional[str] = None) -> str:
    """
    Resolve local path to an asset file or dataset directory, downloading from repository if missing (e.g. in Colab).

    Parameters
    ----------
    filename : str, default=""
        Basename or relative path of the file (e.g. 'puppy.jpeg', 'Advertising.csv').
        If empty string, resolves the root path of the specified asset category directory.
    category : Optional[str], default=None
        Asset directory under 'shared/' ('images' or 'data').
        If None, automatically inferred from file extension or defaults to 'data'.

    Returns
    -------
    str
        Absolute local filesystem path to the asset file or directory.
    """
    category = category or (_category(filename) if filename else "data")
    # In the repository this is '<repo>/shared/<category>'; in Colab it is created under the working directory.
    shared_dir = _find_shared_dir(category) or os.path.join(os.getcwd(), "shared", category)

    if filename == "":
        os.makedirs(shared_dir, exist_ok=True)
        return shared_dir

    clean_rel = filename.replace("\\", "/").lstrip("/")
    if clean_rel.startswith(f"shared/{category}/"):
        clean_rel = clean_rel[len(f"shared/{category}/"):]
    elif clean_rel.startswith(f"{category}/"):
        clean_rel = clean_rel[len(f"{category}/"):]

    candidates = [
        os.path.join(shared_dir, clean_rel),
        os.path.join(os.getcwd(), category, clean_rel),
        os.path.join(os.getcwd(), clean_rel),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    target_path = os.path.join(shared_dir, clean_rel)
    url = get_url(clean_rel, category=category)
    try:
        _download(url, target_path)
        return os.path.abspath(target_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Asset '{filename}' not found locally and remote download failed from: {url}. Error: {exc}"
        ) from exc


DOWNLOAD_TIMEOUT: int = 60  # seconds without any data before a download is aborted


def _download(url: str, target_path: str) -> None:
    """
    Download url to target_path. The data is written to a temporary file that is renamed only
    after a complete download, so an interrupted download never leaves a partial file behind.
    """
    import shutil
    import tempfile
    import urllib.request

    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp, urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            shutil.copyfileobj(response, tmp)
        os.replace(tmp_path, target_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_data_path(filename: str = "") -> str:
    """
    Convenience resolver for datasets in 'shared/data/'.

    Parameters
    ----------
    filename : str, default=""
        Basename of the dataset file (e.g. 'Advertising.csv').
        If empty string, resolves the root path of the data directory.

    Returns
    -------
    str
        Absolute local filesystem path to the dataset.
    """
    return get_file(filename, category="data")


def setup_theme(
    style: str = "whitegrid",
    palette: str = "deep",
    figsize: Tuple[float, float] = (8.0, 5.0),
    font_scale: float = 1.05,
) -> None:
    """
    Configure course Matplotlib and Seaborn aesthetics.

    Parameters
    ----------
    style : str, default="whitegrid"
        Seaborn style theme ('whitegrid', 'ticks', 'white').
    palette : str, default="deep"
        Color palette family.
    figsize : Tuple[float, float], default=(8.0, 5.0)
        Default figure dimensions.
    font_scale : float, default=1.05
        Font scaling multiplier.
    """
    sns.set_theme(
        style=style,
        palette=palette,
        font_scale=font_scale,
        rc={
            "figure.figsize": figsize,
            "figure.dpi": 100,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
        },
    )


# ---------------------------------------------------------------------------
# 2. Base & Specialized Plotter Classes
# ---------------------------------------------------------------------------

class BasePlotter:
    """Base visualizer providing standardized canvas lifecycle and aesthetics."""

    @classmethod
    def canvas(
        cls,
        figsize: Tuple[float, float] = (8.0, 5.0),
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        grid: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Create or configure a standard Matplotlib canvas."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        if title:
            ax.set_title(title, fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.5)
        return fig, ax

    create_canvas = canvas

    @classmethod
    def finalize(
        cls,
        fig: Figure,
        ax: Optional[Axes] = None,
        show: bool = True,
        tight_layout: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """
        Apply layout adjustments and display or return canvas handles.

        `ax` is optional because multi-panel figures have no single axes object.
        """
        if tight_layout:
            fig.tight_layout()
        if show:
            plt.show()
            return None
        return fig, ax

    @staticmethod
    def to_numpy(data: Any, ravel: bool = False) -> np.ndarray:
        """
        Convert PyTorch tensor (with CPU/detach handling), pandas Series/DataFrame,
        NumPy array, or generic sequence to a NumPy ndarray.
        """
        if data is None:
            return np.array([])
        if hasattr(data, "detach"):
            arr = data.detach().cpu().numpy()
        elif hasattr(data, "to_numpy"):
            arr = data.to_numpy()
        else:
            arr = np.asarray(data)
        return arr.ravel() if ravel else arr

    @staticmethod
    def to_scalar(val: Any) -> float:
        """Convert a scalar tensor, numpy scalar, or numeric value to standard Python float."""
        if hasattr(val, "item"):
            return float(val.item())
        return float(val)

    @classmethod
    def ensure_2d(cls, data: Any) -> np.ndarray:
        """Ensure input array has 2 dimensions (e.g. converting (N,) to (N, 1))."""
        arr = cls.to_numpy(data)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    @classmethod
    def eval_predictor(
        cls,
        predictor: Callable[[Any], Any] | Any,
        X: np.ndarray,
        as_tensor_like: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Evaluate predictor on NumPy input X, automatically handling PyTorch tensors,
        device placement, and sklearn-compatible .predict methods, returning a NumPy array.
        """
        import torch

        predict_fn = getattr(predictor, "predict", predictor)
        if as_tensor_like is not None and hasattr(as_tensor_like, "detach"):
            device = getattr(as_tensor_like, "device", "cpu")
            tensor_in = torch.as_tensor(X, dtype=torch.float32, device=device)
            out = predict_fn(tensor_in)
        else:
            try:
                out = predict_fn(X)
            except Exception:
                tensor_in = torch.as_tensor(X, dtype=torch.float32)
                out = predict_fn(tensor_in)
        return cls.to_numpy(out)

    @classmethod
    def get_boundaries(
        cls,
        X: Any,
        padding: float = 0.5,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute padded (min, max) bounds for 2D coordinates."""
        arr = cls.to_numpy(X)
        xlim = (float(arr[:, 0].min() - padding), float(arr[:, 0].max() + padding))
        ylim = (float(arr[:, 1].min() - padding), float(arr[:, 1].max() + padding))
        return xlim, ylim

    @classmethod
    def create_2d_grid(
        cls,
        X: Any,
        padding: float = 0.5,
        resolution: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 2D coordinate meshgrid and stacked evaluation points based on data domain.
        Returns: (xm, ym, grid_pts) where grid_pts has shape (resolution*resolution, 2).
        """
        xlim, ylim = cls.get_boundaries(X, padding=padding)
        xm, ym = np.meshgrid(
            np.linspace(xlim[0], xlim[1], resolution),
            np.linspace(ylim[0], ylim[1], resolution),
        )
        grid_pts = np.c_[xm.ravel(), ym.ravel()]
        return xm, ym, grid_pts

    @classmethod
    def add_annotation_badge(
        cls,
        ax: Axes,
        badge_text: str,
        xy: Tuple[float, float] = (0.05, 0.90),
        fontsize: int = 10,
    ) -> None:
        """Render a rounded text badge in axes fraction coordinates."""
        ax.text(
            xy[0],
            xy[1],
            badge_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#cccccc"),
        )



class RegressionPlotter(BasePlotter):
    """Visualizer for regression curve fitting and residual analysis."""

    @classmethod
    def plot_residuals(
        cls,
        predict: Any,
        X: torch.Tensor,
        y: torch.Tensor,
        show: bool = False,
        xlabel: str = "Input Feature ($x$)",
        ylabel: str = "Target ($t$)",
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.5, 4.0),
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plot regression curve and vertical residual drop-lines.

        Note: Mathematical error metrics (TSS, RSS, ESS, MSE, MAE) are evaluated
        visibly in the calling notebook cell and are NOT calculated here.
        """
        X_flat = cls.to_numpy(X, ravel=True)
        y_flat = cls.to_numpy(y, ravel=True)
        X_eval = X_flat.reshape(-1, 1)
        est_flat = cls.eval_predictor(predict, X_eval, as_tensor_like=X).ravel()

        fig, ax = cls.canvas(figsize=figsize, xlabel=xlabel, ylabel=ylabel)
        effective_title = title or "Regression Fit and Residuals"
        ax.set_title(effective_title, fontsize=12)

        ax.vlines(
            X_flat,
            y_flat,
            est_flat,
            colors="firebrick",
            linestyles="--",
            lw=1.5,
            alpha=0.7,
            label="Residuals",
        )
        ax.scatter(X_flat, y_flat, color="royalblue", s=40, alpha=0.85, label="Observations")
        ax.scatter(X_flat, est_flat, color="forestgreen", s=30, zorder=3, label="Fitted Points")

        x_line = np.linspace(float(X_flat.min()), float(X_flat.max()), 500)
        pred_line_arr = cls.eval_predictor(predict, x_line.reshape(-1, 1), as_tensor_like=X).ravel()

        ax.plot(x_line, pred_line_arr, color="forestgreen", lw=2, label="Model $y(x)$")
        ax.legend(loc="best", fontsize=9.0)

        return cls.finalize(fig, ax, show=show)


    @classmethod
    def plot_fitted_curves(
        cls,
        x: Sequence[float],
        t: Sequence[float],
        x_grid: Sequence[float],
        curves: Dict[str, Sequence[float]],
        points: Optional[Dict[str, Tuple[float, float]]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        colors: Optional[Dict[str, str]] = None,
        linestyles: Optional[Dict[str, str]] = None,
        figsize: Tuple[float, float] = (7.5, 4.5),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Scatter observations and overlay model curves pre-computed on x_grid; optionally mark single points."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        ax.scatter(cls.to_numpy(x, ravel=True), cls.to_numpy(t, ravel=True), color="royalblue", alpha=0.45, s=16, label="Observations")
        for label, values in curves.items():
            ax.plot(cls.to_numpy(x_grid, ravel=True), cls.to_numpy(values, ravel=True), lw=2.2, label=label,
                    color=(colors or {}).get(label), linestyle=(linestyles or {}).get(label, "-"))
        for label, (px, py) in (points or {}).items():
            ax.plot(px, py, marker="o", markersize=9, color="forestgreen", linestyle="none", label=label)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9.5, loc="best")
        return cls.finalize(fig, ax, show=show)

class BoundaryPlotter(BasePlotter):
    """Visualizer for 2D decision boundaries and surfaces."""

    @classmethod
    def plot_binary(
        cls,
        predictor: Callable[[Any], Any],
        X: torch.Tensor,
        y: torch.Tensor,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.5, 6.5),
        show: bool = True,
    ) -> Optional[Tuple[Figure, Axes]]:
        """Plot 2D decision boundary meshgrid and sample scatter points."""
        X_np = cls.to_numpy(X)
        y_np = cls.to_numpy(y, ravel=True)

        xm, ym, grid_pts = cls.create_2d_grid(X_np, padding=0.5, resolution=200)
        p_arr = cls.eval_predictor(predictor, grid_pts, as_tensor_like=X).reshape(xm.shape)

        fig, ax = cls.canvas(figsize=figsize, xlabel="$x_1$", ylabel="$x_2$")
        effective_title = title or "Binary Classification Decision Surface"
        ax.set_title(effective_title, fontsize=12)

        ax.contourf(xm, ym, p_arr, alpha=0.3, cmap="coolwarm")
        ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="coolwarm", edgecolors="k", s=80, lw=1.5)

        return cls.finalize(fig, ax, show=show)


    CLASS_COLORS = ("royalblue", "firebrick", "forestgreen", "darkorange", "purple", "gray")

    @classmethod
    def plot_decision_surface(
        cls,
        X: Any = None,
        t: Any = None,
        xx: Any = None,
        yy: Any = None,
        values: Any = None,
        levels: Optional[Sequence[float]] = None,
        boundaries: Optional[Dict[str, Sequence[float]]] = None,
        class_names: Optional[Dict[Any, str]] = None,
        highlight: Optional[Sequence[int]] = None,
        title: Optional[str] = None,
        xlabel: str = "$x_1$",
        ylabel: str = "$x_2$",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (6.5, 5.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Data coloured by class over pre-computed grid values: class labels as regions, or probabilities with
        contour lines at `levels`; optional linear boundaries {label: (w0, w1, w2)} drawn as w0 + w1 x1 + w2 x2 = 0."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=False)
        t_np = cls.to_numpy(t, ravel=True) if t is not None else np.array([])
        v_np = cls.to_numpy(values) if values is not None else None
        discrete = v_np is not None and levels is None
        labels = sorted(set(np.unique(t_np).tolist()) | (set(np.unique(v_np).tolist()) if discrete else set()))
        color = {lab: cls.CLASS_COLORS[i % len(cls.CLASS_COLORS)] for i, lab in enumerate(labels)}
        names = class_names or {}
        if v_np is not None:
            xx_np, yy_np = cls.to_numpy(xx), cls.to_numpy(yy)
            if discrete:  # one colour per class label
                index = np.vectorize({lab: i for i, lab in enumerate(labels)}.get)(v_np)
                cmap = ListedColormap([color[lab] for lab in labels])
                ax.contourf(xx_np, yy_np, index, levels=np.arange(len(labels) + 1) - 0.5, cmap=cmap, alpha=0.22)
            else:  # probabilities: shading and labelled contour lines
                ax.contourf(xx_np, yy_np, v_np, levels=np.linspace(0.0, 1.0, 21), cmap="coolwarm", alpha=0.35)
                lines = ax.contour(xx_np, yy_np, v_np, levels=list(levels), colors="black", linewidths=1.2)
                ax.clabel(lines, fmt="%.1f", fontsize=9)
            xlim = xlim or (float(xx_np.min()), float(xx_np.max()))
            ylim = ylim or (float(yy_np.min()), float(yy_np.max()))
        if X is not None:
            X_np = cls.to_numpy(X)
            if xlim is None or ylim is None:
                bx, by = cls.get_boundaries(X_np, padding=0.5)
                xlim, ylim = xlim or bx, ylim or by
            for lab in np.unique(t_np):
                mask = t_np == lab
                ax.scatter(X_np[mask, 0], X_np[mask, 1], color=color[lab.item()], edgecolors="k", s=55, lw=0.8,
                           label=names.get(lab.item(), f"$t = {lab.item():g}$"), zorder=3)
            for n in highlight or []:
                ax.scatter([X_np[n, 0]], [X_np[n, 1]], s=260, facecolors="none", edgecolors="black", lw=2.0, zorder=4)
        for (label, w), style in zip((boundaries or {}).items(), ["--", "-", ":", "-."] * 3):
            w0, w1, w2 = (float(v) for v in cls.to_numpy(w, ravel=True)[:3])
            if abs(w2) > 1e-12:
                x1 = np.linspace(xlim[0], xlim[1], 200)
                ax.plot(x1, -(w0 + w1 * x1) / w2, color="black", linestyle=style, lw=2.0, label=label)
            elif abs(w1) > 1e-12:
                ax.axvline(-w0 / w1, color="black", linestyle=style, lw=2.0, label=label)
        handles, _ = ax.get_legend_handles_labels()
        if discrete and X is None:  # regions only: one legend entry per named region
            handles += [Patch(color=color[lab], alpha=0.4, label=names[lab]) for lab in labels if lab in names]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if handles:
            ax.legend(handles=handles, fontsize=9, loc="best")
        return cls.finalize(fig, ax, show=show)



class DiagnosticPlotter(BasePlotter):
    """Visualizer for statistical correlations, pairplots, and diagnostics."""

    @classmethod
    def plot_correlation_matrix(
        cls,
        df: pd.DataFrame,
        title: str = "Empirical Correlation Heatmap",
        figsize: Tuple[float, float] = (7.5, 5.5),
        cmap: str = "vlag",
        show: bool = True,
    ) -> Optional[Tuple[Figure, Axes]]:
        """Render correlation matrix heatmap for numeric DataFrame columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = cls.canvas(figsize=figsize, title=title, grid=False)
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_pairgrid(
        cls,
        df: pd.DataFrame,
        cols: Optional[Sequence[str]] = None,
        hue: Optional[str] = None,
        annotate_pearson: bool = True,
        figsize: Tuple[float, float] = (7.5, 6.5),
        show: bool = True,
    ) -> Optional[sns.PairGrid]:
        """Render styled PairGrid with scatter, KDE/histogram, and Pearson annotations (returns it only if not shown)."""
        data = df[list(cols)] if cols is not None else df
        g = sns.PairGrid(data, hue=hue, height=figsize[1] / max(1, data.shape[1]))
        g.map_upper(sns.scatterplot, s=25, alpha=0.7)
        g.map_diag(sns.histplot, kde=True)
        if annotate_pearson:
            g.map_lower(cls.annotate)
        else:
            g.map_lower(sns.kdeplot, fill=True, alpha=0.4)
        g.fig.set_size_inches(*figsize)
        g.fig.tight_layout()
        if show:
            plt.show()
            return None
        return g

    @classmethod
    def annotate(cls, first_arg: Any, second_arg: Any, **kwargs: Any) -> None:
        """
        Polymorphic annotation helper for:
        1. CNN Mode: (im, ax) annotates pixel values on image patches.
        2. Badge Mode: (ax, text) renders text badge on axes.
        3. PairGrid Mode: (x, y) computes and annotates Pearson correlation r via map_lower.
        """
        if hasattr(second_arg, "text") and hasattr(first_arg, "shape") and len(first_arg.shape) >= 2:
            im = first_arg
            ax = second_arg
            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    val = cls.to_scalar(im[i, j])
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="r",
                        fontsize=kwargs.get("fontsize", 12),
                        weight="bold",
                    )
            return

        if hasattr(first_arg, "text") and callable(getattr(first_arg, "text")):
            ax = first_arg
            text = str(second_arg)
            xy = kwargs.get("xy", (0.05, 0.90))
            fontsize = kwargs.get("fontsize", 10)
            color = kwargs.get("color", "black")
            ax.text(
                xy[0],
                xy[1],
                text,
                transform=ax.transAxes,
                fontsize=fontsize,
                color=color,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
            )
            return

        x, y = first_arg, second_arg
        ax = plt.gca()
        from scipy import stats

        mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(mask) > 1:
            r, _ = stats.pearsonr(x[mask], y[mask])
            ax.annotate(
                f"Pearson $r = {r:.2f}$",
                xy=(0.05, 0.90),
                xycoords="axes fraction",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
            )


class ClassificationPlotter(BasePlotter):
    """Visualizer for classifier evaluation: confusion matrices and ROC curves from pre-computed values."""

    @classmethod
    def plot_confusion_matrix(
        cls,
        cm: Any,
        labels: Optional[Sequence[Any]] = None,
        title: str = "Confusion Matrix",
        figsize: Tuple[float, float] = (6.5, 5.5),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Annotated heatmap of a pre-computed confusion matrix (rows: true class $t$, columns: predicted class)."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax)
        counts = cls.to_numpy(cm)
        names = list(labels) if labels is not None else list(range(counts.shape[0]))
        df_cm = pd.DataFrame(
            counts,
            index=pd.Index(names, name="True Label $t$"),
            columns=pd.Index(names, name=r"Predicted $\hat{t}$"),
        )
        sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="d", annot_kws={"size": 14}, ax=ax)
        ax.set_title(title, fontsize=12)
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_roc_curves(
        cls,
        curves: Dict[str, Tuple[Sequence[float], Sequence[float]]],
        points: Optional[Dict[str, Tuple[float, float]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5.2, 4.6),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """ROC curves from pre-computed (false positive rate, true positive rate) pairs, with the chance diagonal
        and optional marked operating points {label: (FPR, TPR)}."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel="False positive rate (1 - specificity)",
                             ylabel="True positive rate (sensitivity)")
        for label, (fpr, tpr) in curves.items():
            ax.plot(cls.to_numpy(fpr, ravel=True), cls.to_numpy(tpr, ravel=True), lw=2.2, label=label)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.0, label="chance")
        for label, (fpr, tpr) in (points or {}).items():
            ax.plot([fpr], [tpr], marker="o", markersize=9, linestyle="none", color="firebrick", zorder=5, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.legend(loc="lower right", fontsize=9.5)
        return cls.finalize(fig, ax, show=show)


class ImagePlotter(BasePlotter):
    """Visualizer for 2D/3D images and SVD low-rank reconstructions."""

    @classmethod
    def load_image(
        cls,
        name: str,
        as_tensor: bool = False,
        grayscale: bool = False,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Load an image asset from shared/images/.

        Parameters
        ----------
        name : str
            Basename of the image file (e.g. 'puppy.jpeg').
        as_tensor : bool, default=False
            If True, returns a float32 torch.Tensor scaled to [0, 1].
        grayscale : bool, default=False
            If True, converts image to single-channel grayscale.
        """
        from PIL import Image

        path = get_file(name, category="images")
        img = Image.open(path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        if as_tensor:
            import torch

            arr = np.array(img, dtype=np.float32) / 255.0
            if grayscale:
                return torch.from_numpy(arr).unsqueeze(0)
            return torch.from_numpy(arr).permute(2, 0, 1)
        return img

    @classmethod
    def plot_image(
        cls,
        img: Any,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6.0, 6.0),
        cmap: Optional[str] = None,
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Axes]]:
        arr = cls.to_numpy(img)
        if arr.ndim == 3 and arr.shape[0] in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = arr.squeeze(axis=2)


        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if arr.ndim == 2:
            effective_cmap = cmap or "gray"
            ax.imshow(arr, cmap=effective_cmap)
        else:
            ax.imshow(np.clip(arr, 0.0, 1.0) if arr.dtype in (np.float32, np.float64) else arr)

        if title:
            ax.set_title(title, fontsize=12)
        ax.axis("off")
        ax.grid(False)

        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_image_grid(
        cls,
        images: Any,
        titles: Optional[Sequence[str]] = None,
        n_cols: int = 4,
        cmap: str = "gray",
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Show a stack of 2D images (e.g. CNN feature maps) in a grid; receives pre-computed images."""
        arr = cls.to_numpy(images)
        n_rows = (len(arr) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (2.2 * n_cols, 2.2 * n_rows), squeeze=False)
        for i, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if i < len(arr):
                ax.imshow(arr[i], cmap=cmap)
                if titles is not None:
                    ax.set_title(titles[i], fontsize=10)
        return cls.finalize(fig, show=show)

    @classmethod
    def compress_image_svd(
        cls,
        img: Any,
        comps: int = 15,
        std_pca_fn: Optional[Callable[..., Any]] = None,
        figsize: Tuple[float, float] = (12.0, 3.5),
    ) -> None:
        """Demonstrate low-rank image reconstruction via SVD across RGB channels."""
        img_arr = np.asarray(img, dtype=np.float32)
        channels: List[np.ndarray] = []
        colors = ["red", "green", "blue"]

        fig, ax = plt.subplots(1, 3, sharex=True, figsize=figsize)
        for i, c in enumerate(colors):
            channel = img_arr[:, :, i]
            if std_pca_fn is not None:
                import torch

                ch_in = torch.tensor(channel, dtype=torch.float32)
                sc, cp, ve, _ = std_pca_fn(ch_in, k=comps)
                ch_recon = np.dot(np.asarray(sc), np.asarray(cp))
                ve_arr = np.asarray(ve, dtype=float)
            else:
                U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                ch_recon = np.dot(U[:, :comps] * S[:comps], Vt[:comps, :])
                ve_arr = S ** 2

            ve_sum = float(ve_arr.sum())
            ve_norm = ve_arr[:comps] / ve_sum if ve_sum > 0 else ve_arr[:comps]
            ax[i].bar(range(1, comps + 1), ve_norm, color=c)
            ax[i].set_title(f"{c.capitalize()} Channel Variance", fontsize=11)
            ax[i].set_xlabel("Component", fontsize=10.5)

            ch_min, ch_max = float(ch_recon.min()), float(ch_recon.max())
            ch_norm = (ch_recon - ch_min) / (ch_max - ch_min) if ch_max > ch_min else ch_recon
            channels.append(ch_norm)

        fig.tight_layout()
        plt.show()

        recon = np.stack(channels, axis=-1)
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.5))
        axs[0].imshow(img)
        axs[0].set_title("Original Image", fontsize=12)
        axs[0].axis("off")
        axs[0].grid(False)
        axs[1].imshow(recon)
        axs[1].set_title(f"Reconstructed ({comps} components)", fontsize=12)
        axs[1].axis("off")
        axs[1].grid(False)
        fig.tight_layout()
        plt.show()


class CurvePlotter(BasePlotter):
    """Visualizer for pre-computed curves over a shared axis (errors over flexibility, losses, paths)."""

    @classmethod
    def plot_curves(
        cls,
        x: Sequence[float],
        curves: Dict[str, Sequence[float]],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        logx: bool = False,
        logy: bool = False,
        mark_x: Optional[float] = None,
        mark_label: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (7.5, 4.2),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Draw pre-computed curves over a shared x axis (errors over flexibility, coefficient paths, losses)."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        x_arr = cls.to_numpy(x, ravel=True)
        marker = "o" if len(x_arr) <= 25 else None
        colors = plt.cm.tab20.colors if len(curves) > 10 else [None] * len(curves)  # distinct colours for many curves
        for (label, values), color in zip(curves.items(), colors):
            ax.plot(x_arr, cls.to_numpy(values, ravel=True), lw=2.0, marker=marker, ms=4, label=label, color=color)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        if mark_x is not None:
            ax.axvline(mark_x, color="gray", linestyle="--", lw=1.2, label=mark_label)
        if len(curves) > 6:
            ax.legend(fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        else:
            ax.legend(fontsize=9.5, loc="best")
        return cls.finalize(fig, ax, show=show)


class ActivationPlotter(BasePlotter):
    """Visualizer for activation functions, output functions and their derivatives."""

    @classmethod
    def plot_function_panels(
        cls,
        a: Any,
        panels: Sequence[Dict[str, Any]],
        titles: Optional[Sequence[str]] = None,
        ylims: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
        points: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
        dashed: Sequence[str] = (),
        xlabel: str = "Pre-activation $a$",
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Curves pre-computed on the grid a, one panel per dict {label: values}; optional marked points per panel."""
        x = cls.to_numpy(a, ravel=True)
        fig, axes = plt.subplots(1, len(panels), figsize=figsize or (5.25 * len(panels), 4.0), squeeze=False)
        for i, (ax, curves) in enumerate(zip(axes[0], panels)):
            for label, values in curves.items():
                ax.plot(x, cls.to_numpy(values, ravel=True), lw=2.2, ls="--" if label in dashed else "-", label=label)
            if points is not None and points[i] is not None:  # one (x, y) pair or a list of pairs
                marked = np.asarray(points[i], dtype=float).reshape(-1, 2)
                ax.scatter(marked[:, 0], marked[:, 1], color="crimson", s=50, zorder=5)
            ax.axhline(0, color="black", lw=0.8, ls=":")
            ax.axvline(0, color="black", lw=0.8, ls=":")
            ax.set_xlabel(xlabel, fontsize=11)
            if titles is not None:
                ax.set_title(titles[i], fontsize=12)
            if ylims is not None and ylims[i] is not None:
                ax.set_ylim(*ylims[i])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", fontsize=9.5)
        return cls.finalize(fig, show=show)


class PCAPlotter(BasePlotter):
    """Visualizer for PCA: projections, principal directions, explained variance, loadings and biplots."""

    @classmethod
    def plot_projection(
        cls,
        X: Any,
        X_proj: Any,
        direction: Sequence[float],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5.5, 5.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Scatter 2D data, the line along a direction through the origin and the drop lines to the pre-computed projections."""
        X_arr, P_arr = cls.to_numpy(X), cls.to_numpy(X_proj)
        u = cls.to_numpy(direction, ravel=True)
        u = u / np.linalg.norm(u)
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        reach = 1.1 * float(np.abs(X_arr).max())
        ax.plot([-reach * u[0], reach * u[0]], [-reach * u[1], reach * u[1]], color="firebrick", lw=2, label="direction $\\mathbf{u}$")
        for x_n, p_n in zip(X_arr, P_arr):
            ax.plot([x_n[0], p_n[0]], [x_n[1], p_n[1]], color="gray", linestyle="--", lw=0.8, alpha=0.6)
        ax.scatter(X_arr[:, 0], X_arr[:, 1], color="royalblue", alpha=0.7, s=24, label="data $\\mathbf{x}_n$")
        ax.scatter(P_arr[:, 0], P_arr[:, 1], color="black", marker="x", s=24, label="projections")
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.legend(fontsize=9.5, loc="best")
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_explained_variance(
        cls,
        ratio: Sequence[float],
        cumulative_ratio: Sequence[float],
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6.5, 4.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Scree plot: bars of the pre-computed explained-variance ratios and a line of their cumulative sums."""
        r, c = cls.to_numpy(ratio, ravel=True), cls.to_numpy(cumulative_ratio, ravel=True)
        i = np.arange(1, len(r) + 1)
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel="Principal component $i$", ylabel="Proportion of variance")
        ax.bar(i, r, color="royalblue", alpha=0.8, label="explained by component $i$")
        ax.plot(i, c, color="forestgreen", marker="o", lw=2, label="cumulative")
        ax.set_xticks(i)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9.5, loc="center right")
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_directions(
        cls,
        X: Any,
        directions: Dict[str, Sequence[float]],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5.5, 5.0),
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Scatter 2D data and draw pre-computed direction vectors as arrows from the origin."""
        fig, ax = cls.canvas(figsize=figsize, title=title, xlabel=xlabel, ylabel=ylabel)
        X_arr = cls.to_numpy(X)
        ax.scatter(X_arr[:, 0], X_arr[:, 1], color="royalblue", alpha=0.6, s=24)
        for (label, vector), color in zip(directions.items(), ["firebrick", "darkorange", "forestgreen", "purple"]):
            v = cls.to_numpy(vector, ravel=True)
            ax.arrow(0, 0, v[0], v[1], color=color, width=0.02, head_width=0.15, length_includes_head=True)
            ax.plot([], [], color=color, lw=2.5, label=label)
        ax.set_aspect("equal")
        ax.legend(fontsize=9.5, loc="best")
        return cls.finalize(fig, ax, show=show)

    @classmethod
    def plot_loading_heatmap(
        cls,
        loadings: Any,
        x_labels: Optional[Sequence[str]] = None,
        y_labels: Optional[Sequence[str]] = None,
        title: str = "Principal Component Loadings",
        figsize: Tuple[float, float] = (8, 4.2),
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Render styled diverging heatmap for principal component loading matrices."""
        loadings_arr = cls.to_numpy(loadings)
        fig, ax = cls.canvas(figsize=figsize, grid=False)
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
            ax=ax,
        )
        ax.set_title(title, fontsize=12.5)
        return cls.finalize(fig, show=show)

    @classmethod
    def plot_pca_biplot(
        cls,
        z1: int,
        z2: int,
        sc: Any,
        comps: Any,
        obs: Optional[Sequence[Any]],
        features: Sequence[str],
        colors: Sequence[str],
        groups: Optional[Sequence[Any]] = None,
        figsize: Tuple[float, float] = (9.5, 8.5),
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Render PCA biplot: scores as points (labelled by obs, or coloured by groups) and loading vectors as arrows."""
        x = cls.to_numpy(sc)[:, z1]
        y = cls.to_numpy(sc)[:, z2]
        comps_arr = cls.to_numpy(comps)

        fig, ax = cls.canvas(figsize=figsize)
        ax.set_xlabel(f"$z_{{{z1 + 1}}}$", fontsize=12.5)
        ax.set_ylabel(f"$z_{{{z2 + 1}}}$", fontsize=12.5)

        sx = float((x.max() - x.min()) / 2)
        sy = float((y.max() - y.min()) / 2)

        if groups is None:
            ax.scatter(x, y, alpha=0.5, color="royalblue")
            for i in range(len(obs)):
                ax.text(x[i], y[i], str(obs[i]), ha="center", fontsize=10, alpha=0.75)
        else:
            g = np.asarray(groups)
            for label in pd.unique(g):
                ax.scatter(x[g == label], y[g == label], alpha=0.6, s=28, label=str(label))
            ax.legend(fontsize=10, loc="best")

        vec = comps_arr[[z1, z2], :].T
        for i in range(len(vec)):
            c = colors[i % len(colors)]
            ax.arrow(
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
            ax.text(
                vec[i, 0] * sx * 1.15,
                vec[i, 1] * sy * 1.15,
                features[i],
                color=c,
                fontsize=11.5,
                fontweight="bold",
            )

        ax.set_title(f"Principal Component Biplot: $z_{{{z1 + 1}}}$ vs $z_{{{z2 + 1}}}$", fontsize=13)
        return cls.finalize(fig, show=show)


# ---------------------------------------------------------------------------
# 3. Public Functional Facades (100% Backward Compatibility)
# ---------------------------------------------------------------------------

plot_residuals = RegressionPlotter.plot_residuals
plot_fitted_curves = RegressionPlotter.plot_fitted_curves
plot_curves = CurvePlotter.plot_curves
plot_binary = BoundaryPlotter.plot_binary
plot_decision_surface = BoundaryPlotter.plot_decision_surface
plot_correlation_matrix = DiagnosticPlotter.plot_correlation_matrix
plot_pairgrid = DiagnosticPlotter.plot_pairgrid
plot_confusion_matrix = ClassificationPlotter.plot_confusion_matrix
plot_roc_curves = ClassificationPlotter.plot_roc_curves
annotate = DiagnosticPlotter.annotate
load_image = ImagePlotter.load_image
plot_image = ImagePlotter.plot_image
plot_image_grid = ImagePlotter.plot_image_grid
compress_image_svd = ImagePlotter.compress_image_svd
plot_function_panels = ActivationPlotter.plot_function_panels
plot_projection = PCAPlotter.plot_projection
plot_explained_variance = PCAPlotter.plot_explained_variance
plot_directions = PCAPlotter.plot_directions
plot_loading_heatmap = PCAPlotter.plot_loading_heatmap
plot_pca_biplot = PCAPlotter.plot_pca_biplot

# Data and Predictor Conversion Facades
to_numpy = BasePlotter.to_numpy
to_scalar = BasePlotter.to_scalar
ensure_2d = BasePlotter.ensure_2d
eval_predictor = BasePlotter.eval_predictor
get_boundaries = BasePlotter.get_boundaries
create_2d_grid = BasePlotter.create_2d_grid

