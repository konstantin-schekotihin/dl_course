"""
Canonical Shared Course Helpers & Infrastructure (dl_course / shared).

Provides unified asset resolution, aesthetics, image utilities, and diagnostic visualizers.
Inherited and re-exported by course-specific helper modules across ML-DL and AI-ML.

Modules:
1. Course Aesthetics & Asset Resolution (get_file, get_url, get_data_path, setup_theme, print_math)
2. Base & Specialized Plotter Classes (BasePlotter, RegressionPlotter, BoundaryPlotter, DiagnosticPlotter, ImagePlotter)
3. Image Loading, Processing & Display (load_image, plot_image, compress_image_svd)
4. Functional Facades (plot_residuals, plot_binary, plot_correlation_matrix, plot_pairgrid, annotate)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
        figsize: Tuple[float, float] = (7.5, 4.5),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Optional[Tuple[Figure, Optional[Axes]]]:
        """Scatter observations and overlay model curves pre-computed on x_grid; optionally mark single points."""
        fig, ax = cls.canvas(figsize=figsize, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        ax.scatter(cls.to_numpy(x, ravel=True), cls.to_numpy(t, ravel=True), color="royalblue", alpha=0.45, s=16, label="Observations")
        for label, values in curves.items():
            ax.plot(cls.to_numpy(x_grid, ravel=True), cls.to_numpy(values, ravel=True), lw=2.2, label=label)
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
    ) -> sns.PairGrid:
        """Render styled PairGrid with scatter, KDE/histogram, and Pearson annotations."""
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
        plt.show()
        return g

    @classmethod
    def plot_correlation_matrix_and_pairgrid(
        cls,
        data: Any,
        figsize: Tuple[float, float] = (7.5, 5.5),
    ) -> None:
        """Composite visualizer rendering correlation heatmap followed by PairGrid."""
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        cls.plot_correlation_matrix(df, figsize=figsize, show=True)
        cls.plot_pairgrid(df, figsize=figsize)

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


# ---------------------------------------------------------------------------
# 3. Public Functional Facades (100% Backward Compatibility)
# ---------------------------------------------------------------------------

plot_residuals = RegressionPlotter.plot_residuals
plot_fitted_curves = RegressionPlotter.plot_fitted_curves
plot_binary = BoundaryPlotter.plot_binary
plot_correlation_matrix = DiagnosticPlotter.plot_correlation_matrix
plot_pairgrid = DiagnosticPlotter.plot_pairgrid
plot_correlation_matrix_and_pairgrid = DiagnosticPlotter.plot_correlation_matrix_and_pairgrid
annotate = DiagnosticPlotter.annotate
load_image = ImagePlotter.load_image
plot_image = ImagePlotter.plot_image
compress_image_svd = ImagePlotter.compress_image_svd

# Data and Predictor Conversion Facades
to_numpy = BasePlotter.to_numpy
to_scalar = BasePlotter.to_scalar
ensure_2d = BasePlotter.ensure_2d
eval_predictor = BasePlotter.eval_predictor
get_boundaries = BasePlotter.get_boundaries
create_2d_grid = BasePlotter.create_2d_grid

