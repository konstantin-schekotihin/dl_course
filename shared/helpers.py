"""
Canonical Shared Course Helpers & Infrastructure (dl_course / shared).

Provides unified asset resolution, aesthetics, canonical dataset catalogs,
vision benchmark loaders, image utilities, and shared diagnostic visualizers.
Inherited and re-exported by course-specific helper modules across ML-DL and AI-ML.

Modules:
1. Course Aesthetics & Asset Resolution (get_file, get_url, setup_theme)
2. Canonical Tabular Dataset Catalog (load_dataset)
3. Vision Benchmark Loaders (load_cifar10, load_mnist, load_fashion_mnist)
4. Image Loading, Processing & Display (load_image, plot_image, compress_image_svd)
5. Shared Diagnostics & Exploratory Visualizers (plot_pairgrid, plot_residuals, plot_binary, annotate)
"""

import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image
import scipy.stats as stats
import seaborn as sns
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
import torch
import torch.utils.data

try:
    import torchvision
    import torchvision.datasets as tv_datasets
    import torchvision.transforms as tv_transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


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
    if category is None:
        ext = os.path.splitext(basename)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".tif", ".tiff"}:
            category = "images"
        else:
            category = "data"
    return f"{COURSE_RAW_URL}/{category}/{basename}"


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
        Existing local path to the resolved file or directory.
    """
    if filename and os.path.exists(filename):
        return filename

    basename = os.path.basename(filename) if filename else ""
    if category is None:
        if basename:
            ext = os.path.splitext(basename)[1].lower()
            category = "images" if ext in {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".tif", ".tiff"} else "data"
        else:
            category = "data"

    if basename:
        local_candidates = [
            os.path.join("..", "..", "shared", category, basename),
            os.path.join("..", "shared", category, basename),
            os.path.join("shared", category, basename),
            os.path.join("data", basename) if category == "data" else os.path.join("images", basename),
            basename,
        ]
    else:
        local_candidates = [
            os.path.join("..", "..", "shared", category),
            os.path.join("..", "shared", category),
            os.path.join("shared", category),
            category,
            f"./{category}",
        ]

    for path in local_candidates:
        if os.path.exists(path):
            return path

    if not basename:
        os.makedirs(f"./{category}", exist_ok=True)
        return f"./{category}"

    url = get_url(basename, category)
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CourseAssetDownloader)"})
        with urllib.request.urlopen(req) as response, open(basename, "wb") as out_file:
            out_file.write(response.read())
        return basename
    except Exception as exc:
        raise FileNotFoundError(
            f"Asset '{basename}' not found locally and could not be downloaded from '{url}'. Error: {exc}"
        ) from exc


def get_data_path(filename: str = "") -> str:
    """
    Resolve local dataset file or root directory path with fallback to remote course repository for Colab.
    Backward-compatible alias for get_file(filename, category='data').
    """
    return get_file(filename=filename, category="data")


def setup_theme(
    style: str = "whitegrid",
    palette: str = "deep",
    font_scale: float = 1.0,
    font_family: str = "sans-serif",
    **kwargs: Any,
) -> None:
    """
    Apply unified course aesthetic theme across Seaborn and Matplotlib.
    Standardizes figure DPI, font scaling, grid visibility, and AAU palette.
    """
    aau_palette = ["#003366", "#D9534F", "#5CB85C", "#F0AD4E", "#5BC0DE", "#333333"]
    sns.set_theme(
        style=style,
        palette=palette,
        font=font_family,
        font_scale=font_scale,
        rc={
            "figure.dpi": 100,
            "figure.figsize": (8, 4.2),
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            **kwargs,
        },
    )
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=aau_palette)


# ---------------------------------------------------------------------------
# 2. Canonical Tabular Dataset Catalog (load_dataset)
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    as_tensor: bool = False,
    return_X_y: bool = False,
    target_column: Optional[str] = None,
    **kwargs: Any,
) -> Union[pd.DataFrame, Tuple[Any, Any]]:
    """
    Load canonical course tabular dataset with standardized indexing, column dtypes, and delimiters.

    Supported Datasets:
    -------------------
    - 'advertising'   : Advertising.csv (TV, Radio, Newspaper -> Sales; drops index column)
    - 'usarrests'     : USArrests.csv (State crime statistics; index set to State name)
    - 'default'       : Default.csv (Credit default classification; drops index column)
    - 'breast_cancer' : breast-cancer.csv (Medical classification)
    - 'model_example' : model_example.csv (Budget, Generator, Revenue; handles sep=';')
    - 'iris'          : Fisher's Iris botanical benchmark (scikit-learn)

    Parameters
    ----------
    name : str
        Canonical dataset identifier (case-insensitive, underscores or hyphens accepted).
    as_tensor : bool, default=False
        If True, converts output to PyTorch float32 tensors.
    return_X_y : bool, default=False
        If True, returns a 2-tuple (features, target) instead of a single DataFrame.
    target_column : Optional[str], default=None
        Custom target column name to extract when return_X_y=True.
    **kwargs : Any
        Additional keyword arguments forwarded to pd.read_csv.

    Returns
    -------
    Union[pd.DataFrame, Tuple[Any, Any]]
        Cleaned pandas DataFrame, or (X, y) pair as DataFrames/Series or PyTorch tensors.
    """
    key = name.strip().lower().replace("-", "_")

    if key == "advertising":
        path = get_file("Advertising.csv", category="data")
        df = pd.read_csv(path, **kwargs)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        default_target = "Sales"

    elif key == "usarrests":
        path = get_file("USArrests.csv", category="data")
        df = pd.read_csv(path, **kwargs)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "State"}).set_index("State")
        default_target = None

    elif key == "default":
        path = get_file("Default.csv", category="data")
        df = pd.read_csv(path, **kwargs)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        default_target = "default"

    elif key in {"breast_cancer", "breastcancer"}:
        path = get_file("breast-cancer.csv", category="data")
        df = pd.read_csv(path, **kwargs)
        default_target = "recurrence"

    elif key == "model_example":
        path = get_file("model_example.csv", category="data")
        sep = kwargs.pop("sep", ";")
        df = pd.read_csv(path, sep=sep, **kwargs)
        default_target = "Revenue"

    elif key == "iris":
        iris = sk_datasets.load_iris(as_frame=True)
        df = iris.frame.copy()
        default_target = "target"

    else:
        # Fallback to direct file loading by basename if not in predefined registry
        path = get_file(name if name.endswith(".csv") else f"{name}.csv", category="data")
        df = pd.read_csv(path, **kwargs)
        default_target = None

    if return_X_y:
        tgt = target_column or default_target
        if tgt is None:
            raise ValueError(f"Dataset '{name}' has no default target column. Specify 'target_column'.")
        if tgt not in df.columns:
            raise KeyError(f"Target column '{tgt}' not found in dataset columns: {list(df.columns)}")
        X = df.drop(columns=[tgt])
        y = df[tgt]

        if as_tensor:
            X_num = X.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
            if pd.api.types.is_numeric_dtype(y):
                y_num = y.to_numpy(dtype=np.float32)
            else:
                y_num = pd.Categorical(y).codes.astype(np.float32)
            return torch.from_numpy(X_num), torch.from_numpy(y_num)
        return X, y

    if as_tensor:
        numeric_df = df.select_dtypes(include=[np.number])
        return torch.from_numpy(numeric_df.to_numpy(dtype=np.float32))

    return df


# ---------------------------------------------------------------------------
# 3. Vision Benchmark Loaders (load_cifar10, load_mnist, load_fashion_mnist)
# ---------------------------------------------------------------------------

def load_cifar10(
    batch_size: int = 64,
    train: bool = True,
    download: bool = True,
    root: Optional[str] = None,
    normalize: bool = True,
    shuffle: bool = True,
) -> Tuple[Any, torch.utils.data.DataLoader]:
    """
    Standard PyTorch CIFAR-10 dataset and DataLoader loader.
    Applies canonical RGB channel mean and standard deviation normalization.
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required to load CIFAR-10. Install via 'uv pip install torchvision'.")

    data_dir = root or get_data_path("")
    transforms_list: List[Any] = [tv_transforms.ToTensor()]
    if normalize:
        transforms_list.append(
            tv_transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            )
        )
    transform = tv_transforms.Compose(transforms_list)
    dataset = tv_datasets.CIFAR10(root=data_dir, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


def load_mnist(
    batch_size: int = 64,
    train: bool = True,
    download: bool = True,
    root: Optional[str] = None,
    normalize: bool = True,
    shuffle: bool = True,
) -> Tuple[Any, torch.utils.data.DataLoader]:
    """
    Standard PyTorch MNIST dataset and DataLoader loader.
    Applies single-channel mean (0.1307) and std (0.3081) normalization.
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required to load MNIST. Install via 'uv pip install torchvision'.")

    data_dir = root or get_data_path("")
    transforms_list: List[Any] = [tv_transforms.ToTensor()]
    if normalize:
        transforms_list.append(tv_transforms.Normalize(mean=(0.1307,), std=(0.3081,)))
    transform = tv_transforms.Compose(transforms_list)
    dataset = tv_datasets.MNIST(root=data_dir, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


def load_fashion_mnist(
    batch_size: int = 64,
    train: bool = True,
    download: bool = True,
    root: Optional[str] = None,
    normalize: bool = True,
    shuffle: bool = True,
) -> Tuple[Any, torch.utils.data.DataLoader]:
    """
    Standard PyTorch FashionMNIST dataset and DataLoader loader.
    Applies single-channel mean (0.2860) and std (0.3530) normalization.
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required to load FashionMNIST. Install via 'uv pip install torchvision'.")

    data_dir = root or get_data_path("")
    transforms_list: List[Any] = [tv_transforms.ToTensor()]
    if normalize:
        transforms_list.append(tv_transforms.Normalize(mean=(0.2860,), std=(0.3530,)))
    transform = tv_transforms.Compose(transforms_list)
    dataset = tv_datasets.FashionMNIST(root=data_dir, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


# ---------------------------------------------------------------------------
# 4. Image Loading, Processing & Display (load_image, plot_image, compress_image_svd)
# ---------------------------------------------------------------------------

def load_image(
    name: str,
    as_tensor: bool = False,
    grayscale: bool = False,
    normalize: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Load an image from shared/images/ (or remote course repository) as a clean float NumPy array or PyTorch tensor.

    Parameters
    ----------
    name : str
        Basename of the image (e.g. 'puppy.jpeg', 'dog.jpg').
    as_tensor : bool, default=False
        If True, converts to PyTorch tensor with shape [C, H, W] (or [H, W] if grayscale).
    grayscale : bool, default=False
        If True, converts to single-channel grayscale image.
    normalize : bool, default=True
        If True, rescales pixel intensities into the unit float interval [0.0, 1.0].

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Processed image as NumPy array or PyTorch tensor.
    """
    img_path = get_file(name, category="images")
    with Image.open(img_path) as pil_img:
        if grayscale:
            pil_img = pil_img.convert("L")
        else:
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img, dtype=np.float32)

    if normalize:
        arr /= 255.0

    if as_tensor:
        if grayscale:
            t = torch.from_numpy(arr)
        else:
            t = torch.from_numpy(arr).permute(2, 0, 1)
        return t

    return arr


def plot_image(
    img: Any,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
    cmap: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Display a 2D grayscale or 3D RGB image cleanly without axis ticks or grid line artifacts.
    """
    if hasattr(img, "detach"):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = arr.squeeze(axis=2)
    else:
        arr = np.asarray(img)

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
    fig.tight_layout()
    plt.show()


def compress_image_svd(
    img: Any,
    comps: int = 15,
    std_pca_fn: Optional[Callable[..., Any]] = None,
    figsize: Tuple[float, float] = (12, 3.5),
) -> None:
    """
    Demonstrate low-rank image reconstruction via SVD/PCA across RGB channels.
    Renders 3-channel variance bar charts followed by original vs reconstructed comparison.
    """
    img_arr = np.asarray(img, dtype=np.float32)
    channels: List[np.ndarray] = []
    colors = ["red", "green", "blue"]

    fig, ax = plt.subplots(1, 3, sharex=True, figsize=figsize)
    for i, c in enumerate(colors):
        channel = img_arr[:, :, i]
        if std_pca_fn is not None:
            try:
                ch_in = torch.tensor(channel, dtype=torch.float32)
            except Exception:
                ch_in = channel
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
# 5. Shared Diagnostics & Exploratory Visualizers
# ---------------------------------------------------------------------------

def annotate(first_arg: Any, second_arg: Any, **kwargs: Any) -> None:
    """
    Polymorphic annotation helper supporting:
    1. CNN Mode: When called with (im, ax), annotates numeric pixel values onto image patches.
    2. Axes Badge Mode: When called with (ax, text), renders styled text badge on the axes.
    3. PairGrid Mode: When called with (x, y) via map_lower, computes and annotates Pearson correlation r.
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
        ax.text(xy[0], xy[1], text, transform=ax.transAxes, fontsize=fontsize, color=color,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))
        return

    # Seaborn map_lower callback signature: annotate(x, y, **kws)
    x, y = first_arg, second_arg
    ax = plt.gca()
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


def plot_pairgrid(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    hue: Optional[str] = None,
    annotate_pearson: bool = True,
    figsize: Tuple[float, float] = (7.5, 6.5),
) -> sns.PairGrid:
    """
    Render styled PairGrid featuring upper scatter + Pearson correlation badge,
    diagonal univariate KDE/histogram, and lower bivariate KDE contours.
    """
    data = df[list(cols)] if cols is not None else df
    g = sns.PairGrid(data, hue=hue, height=figsize[1] / max(1, data.shape[1]))
    g.map_upper(sns.scatterplot, s=25, alpha=0.7)
    g.map_diag(sns.histplot, kde=True)
    if annotate_pearson:
        g.map_lower(annotate)
    else:
        g.map_lower(sns.kdeplot, fill=True, alpha=0.4)
    g.fig.set_size_inches(*figsize)
    g.fig.tight_layout()
    plt.show()
    return g


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Empirical Correlation Heatmap",
    figsize: Tuple[float, float] = (7.5, 5.5),
    cmap: str = "vlag",
) -> None:
    """Render correlation matrix heatmap for numeric columns in DataFrame."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=figsize)
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
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_correlation_matrix_and_pairgrid(
    data: Any,
    figsize: Tuple[float, float] = (7.5, 5.5),
) -> None:
    """Backward-compatible composite visualizer rendering correlation heatmap and PairGrid."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    plot_correlation_matrix(df, figsize=figsize)
    plot_pairgrid(df, figsize=figsize)


def plot_residuals(
    predict: Union[Callable[[Any], Any], Any],
    X: Any,
    y: Any,
    show: bool = False,
    xlabel: str = "Input Feature ($x$)",
    ylabel: str = "Target ($t$)",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 4.0),
) -> None:
    """
    Plot linear or polynomial regression fit and residual errors (vertical drop-lines).
    Computes and displays TSS, RSS, ESS, MSE, and MAE.
    """
    if hasattr(predict, "predict"):
        predict_fn = predict.predict
    elif callable(predict):
        predict_fn = predict
    else:
        raise TypeError("predict must be a callable or model with a .predict() method.")

    if hasattr(X, "ndim") and X.ndim == 1:
        X_eval = X.reshape(-1, 1)
    elif isinstance(X, (list, tuple)):
        X_eval = np.asarray(X).reshape(-1, 1)
    else:
        X_eval = X

    est = predict_fn(X_eval)
    if hasattr(est, "reshape"):
        est = est.reshape(-1)

    if hasattr(y, "device") and hasattr(est, "device"):
        rss = torch.sum(torch.pow(y - est, 2)).item()
        tss = torch.sum(torch.pow(y - torch.mean(y), 2)).item()
        mae = torch.sum(torch.abs(y - est)).item() / len(est)
        mse = rss / len(est)
        X_np, y_np, est_np = X.cpu().numpy(), y.cpu().numpy(), est.cpu().numpy()
    else:
        rss = float(np.sum(np.power(y - est, 2)))
        tss = float(np.sum(np.power(y - np.mean(y), 2)))
        mae = float(np.sum(np.abs(y - est)) / len(est))
        mse = rss / len(est)
        X_np, y_np, est_np = np.asarray(X), np.asarray(y), np.asarray(est)

    if show:
        display(Markdown(rf"$\mathrm{{TSS}} = {tss:.3f}$ (total sum of squares)"))
        display(Markdown(rf"$\mathrm{{RSS}} = {rss:.3f}$ (residual sum of squares)"))
        display(Markdown(rf"$\mathrm{{ESS}} = \mathrm{{TSS}} - \mathrm{{RSS}} = {tss - rss:.3f}$ (explained sum of squares)"))
        display(Markdown(rf"$\mathrm{{MSE}} = {mse:.3f}$ (mean squared error)"))
        display(Markdown(rf"$\mathrm{{MAE}} = {mae:.3f}$ (mean absolute error)"))

    X_flat = X_np.ravel()
    y_flat = y_np.ravel()
    est_flat = est_np.ravel()
    e_pot = 0.5 * rss

    plt.figure(figsize=figsize)
    plt.vlines(X_flat, y_flat, est_flat, colors="firebrick", linestyles="--", lw=1.5, alpha=0.7, label=r"Residuals ($e_n$)")
    plt.scatter(X_flat, y_flat, color="royalblue", s=40, alpha=0.85, label=r"Observations ($(x_n, t_n)$)")
    plt.scatter(X_flat, est_flat, color="forestgreen", s=30, zorder=3, label=r"Fitted Points ($y(x_n)$)")

    x_line = np.linspace(float(X_flat.min()), float(X_flat.max()), 500)
    if hasattr(X, "device"):
        pred_line = predict_fn(torch.from_numpy(x_line).float().reshape(-1, 1).to(X.device))
        pred_line = pred_line.cpu().numpy().ravel() if hasattr(pred_line, "cpu") else np.asarray(pred_line).ravel()
    else:
        pred_line = np.asarray(predict_fn(x_line.reshape(-1, 1))).ravel()
    plt.plot(x_line, pred_line, color="forestgreen", lw=2, label="Model $y(x)$")

    effective_title = title or f"Physical Spring Analogy: Residual Tension ($E(\\mathbf{{w}}) = {e_pot:.3f}$)"
    plt.title(effective_title, fontsize=11)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.legend(loc="best", fontsize=9.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_binary(
    predictor: Callable[[Any], Any],
    X: Any,
    y: Any,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.5, 6.5),
) -> None:
    """
    Plot 2D decision boundary meshgrid and data points for binary classification models.
    """
    X_np = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
    y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)

    x1_min, x1_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    x2_min, x2_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5

    xm, ym = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200),
    )
    grid_pts = np.c_[xm.ravel(), ym.ravel()]
    if hasattr(X, "device"):
        p = predictor(torch.from_numpy(grid_pts).float().to(X.device))
        p = p.cpu().numpy().reshape(xm.shape) if hasattr(p, "cpu") else np.asarray(p).reshape(xm.shape)
    else:
        p = np.asarray(predictor(grid_pts)).reshape(xm.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xm, ym, p, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np.ravel(), cmap="coolwarm", edgecolors="k", s=80, lw=1.5)
    effective_title = title or "Binary Classification Decision Surface"
    ax.set_title(effective_title, fontsize=12)
    ax.set_xlabel("$x_1$", fontsize=11.5)
    ax.set_ylabel("$x_2$", fontsize=11.5)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
