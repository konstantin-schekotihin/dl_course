"""
Self-contained helper utilities for the AI and Machine Learning course.
Provides standalone plotting and data helpers decoupled from 03_DL.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import scipy.stats as stats
import seaborn as sns
import sklearn.metrics as metrics
import torch

DATA_BASE_URL = os.environ.get(
    "COURSE_DATA_URL",
    "https://raw.githubusercontent.com/konstantin-schekotihin/dl_course/master/shared/data"
)

def get_data_path(filename):
    """Resolve local dataset path with fallback to remote course repository for Colab."""
    local_paths = [
        os.path.join("../../shared/data", filename),
        os.path.join("../shared/data", filename),
        os.path.join("shared/data", filename),
        os.path.join("data", filename),
        filename
    ]
    for p in local_paths:
        if os.path.exists(p):
            return p
    return f"{DATA_BASE_URL}/{filename}"

def setup_theme():
    """Apply unified course-standard Seaborn plotting aesthetics."""
    sns.set_theme(
        style="whitegrid",
        palette="deep",
        rc={
            "figure.figsize": (8, 6),
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

# Automatically configure unified course theme on import
setup_theme()

cm_binary = ListedColormap(['green', 'blue'])

def get_boundaries(X):
    """Compute 2D coordinate plot boundaries with padding."""
    X_arr = np.asarray(X)
    xlim = (float(np.min(X_arr[:, 0] - 1)), float(np.max(X_arr[:, 0] + 1)))
    ylim = (float(np.min(X_arr[:, 1] - 1)), float(np.max(X_arr[:, 1] + 1)))
    return xlim, ylim

def plot_knn_intuition(iris, show=False):
    """Render kNN neighborhood decision radius illustration."""
    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(x=iris.sepal_length, y=iris.petal_length, hue=iris.species)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    if show:
        plt.scatter(x=5.5, y=4.7, color='m')
        ax.add_artist(plt.Circle((5.5, 4.7), radius=0.3, color='m', alpha=0.3))
    plt.show()


def annotate(x, y, **kws):
    """Annotate seaborn pairgrid subplots with Pearson correlation coefficient."""
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}, p = {:.3f} ".format(r, p),
                xy=(.1, 1), xycoords=ax.transAxes)


def plot_residuals(f, pred, resp, show=False):
    """Plot regression line and residuals between predictions and responses."""
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
        rss = np.sum((resp_arr - est_arr) ** 2)
        tss = np.sum((resp_arr - np.mean(resp_arr)) ** 2)
        print("TSS = {:.3f} - total sum of squares".format(tss))
        print("RSS = {:.3f} - residual sum of squares".format(rss))
        print("ESS = TSS - RSS = {:.3f} - explained sum of squares".format(tss - rss))
        print("MSE = {:.3f} - mean squared error".format(rss / len(resp_arr)))
        print("MAE = {:.3f} - mean absolute error".format(np.mean(np.abs(resp_arr - est_arr))))

    l = np.linspace(0, 300, num=1000)
    plt.figure(figsize=(8, 6))
    plt.vlines(pred_arr, resp_arr, est_arr, colors="blue", linestyles="dashed", alpha=0.5, label="Residuals")
    plt.scatter(pred_arr, resp_arr, color="blue", alpha=0.6, label="Observations")
    plt.scatter(pred_arr, est_arr, color="red", s=25, label="Predictions")
    try:
        l_pred = np.asarray(f(l.reshape(-1, 1))).ravel()
    except Exception:
        l_pred = np.asarray(f(l)).ravel()
    plt.plot(l, l_pred, "r-", lw=2, label="Fit")
    plt.xlabel("TV Budget")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()


def plot_linear_fit(X, y, predict_fn):
    """Plot scatter data and regression predictor curve."""
    X_arr = np.asarray(X).ravel()
    y_arr = np.asarray(y).ravel()
    l = np.linspace(X_arr.min() - 0.5, X_arr.max() + 0.5, num=200).reshape(-1, 1)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_arr, y_arr, alpha=0.5, label="Observations")
    plt.plot(l, predict_fn(l), "r-", lw=2, label="Linear fit")
    plt.xlabel("Standardized Balance ($x$)")
    plt.ylabel("Default Class ($y$)")
    plt.legend()
    plt.show()


def plot_positive_vs_log(l, f_vals, log_vals, is_positive=True):
    """Plot positive surrogate function f and its monotonic log transform."""
    plt.figure(figsize=(8, 5))
    plt.plot(l, f_vals, "b-", lw=2, label=r"$f(x)$")
    if not is_positive:
        plt.title("Warning: The function is not strictly positive!")
    elif log_vals is not None:
        plt.plot(l, log_vals, "r--", lw=2, label=r"$\ln f(x)$")
    plt.xlabel("$x$")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_logistic_curve(x_vals, y_vals, w_0, w_1):
    """Plot logistic sigmoid response curve over standardized feature range."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, "b-", lw=2, label=r"$\sigma(%.1f + %.1fx)$" % (w_0, w_1))
    plt.axhline(0.5, color="gray", linestyle="--", label="Decision threshold = 0.5")
    plt.xlabel("Standardized Balance ($x$)")
    plt.ylabel("Predicted Probability ($y$)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.show()


def plot_binary(predictor, X, y):
    """
    Plot decision boundaries for 2D binary classification.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c="w")

    # Plot decision boundaries
    ax = plt.gca()
    y1, y2 = ax.get_ylim()
    x1, x2 = ax.get_xlim()
    xm, ym = np.meshgrid(
        np.arange(x1, x2, (x2 - x1) / 100),
        np.arange(y1, y2, (y2 - y1) / 100)
    )
    p = predictor(np.c_[xm.ravel(), ym.ravel()]).reshape(xm.shape)
    plt.scatter(xm, ym, c=p, cmap="coolwarm", alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="w", s=100, linewidths=2)
    plt.show()


def plot_knn_results(train_X, train_y, test_X, test_y, pred_y, target_names=None, feature_cols=None):
    """
    Plot dual-subplot evaluation for kNN:
    - Left: Confusion matrix heatmap.
    - Right: 2D scatter plot showing training data and test classification errors (faults).
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    cm = metrics.confusion_matrix(test_y, pred_y)
    sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", ax=axs[0], annot_kws={"size": 14})
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel("Predicted Label")
    axs[0].set_ylabel("True Label")

    colors = ["darkblue", "green", "red"]
    data_markers = ["s", "s", "s"]
    markers = ["x", "D", "o"]

    if target_names is None:
        target_names = np.unique(np.concatenate([train_y, test_y]))

    d1, d2 = 3, 0
    train_X_arr = np.asarray(train_X)
    test_X_arr = np.asarray(test_X)

    for n, color in enumerate(colors[:len(target_names)]):
        t_idx = np.where(train_y == target_names[n])[0]
        sns.scatterplot(
            x=train_X_arr[t_idx, d1], y=train_X_arr[t_idx, d2],
            color=color, label="Train: %s" % target_names[n],
            marker=data_markers[n], ax=axs[1], s=48
        )

    for n, color in enumerate(colors[:len(target_names)]):
        for k, marker in enumerate(markers[:len(target_names)]):
            inc_idx = np.where((pred_y != test_y) & (test_y == target_names[n]) & (pred_y == target_names[k]))[0]
            if len(inc_idx) > 0:
                sns.scatterplot(
                    x=test_X_arr[inc_idx, d1], y=test_X_arr[inc_idx, d2],
                    marker=marker, color=color,
                    label="Fault: %s" % target_names[k], ax=axs[1], s=64
                )

    if feature_cols:
        axs[1].set_xlabel(feature_cols[d1] if d1 < len(feature_cols) else "Feature %d" % d1)
        axs[1].set_ylabel(feature_cols[d2] if d2 < len(feature_cols) else "Feature %d" % d2)
    axs[1].legend(loc="upper left")
    axs[1].set_title("Classification Results")
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, ax=None, title="Decision Boundary"):
    """
    Render 2D classification decision surface and scatter observations.
    Supports both standalone rendering (ax=None) and subplot integration.
    """
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    x_min, x_max = X_arr[:, 0].min() - 0.5, X_arr[:, 0].max() + 0.5
    y_min, y_max = X_arr[:, 1].min() - 0.5, X_arr[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
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
            X_arr[mask, 0], X_arr[mask, 1],
            c=palette[idx % len(palette)],
            label="Class %s" % lbl,
            edgecolors="k",
            alpha=0.75,
            s=40
        )

    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="best")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_svm_margins(model, ax=None, xlim=None, ylim=None):
    """
    Plot SVM decision boundary, margin contours, and support vectors.
    """
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(
        XX, YY, Z,
        colors=["blue", "black"],
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"]
    )
    if hasattr(model, "support_vectors_"):
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors="none",
            edgecolors="r",
            s=100,
            linewidths=1.5,
            label="Support Vectors"
        )

    if standalone:
        plt.show()


def plot_pca_scores(X, ort, x_label=None, y_label=None):
    """
    Plot projection of 2D data points onto a PCA orientation vector.
    """
    ort = np.asarray(ort, dtype=np.float32).reshape(2, 1)
    X_arr = np.asarray(X)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    if y_label:
        ax[0].set_ylabel(y_label)

    for a in ax:
        if x_label:
            a.set_xlabel(x_label)
        a.scatter(X_arr[:, 0], X_arr[:, 1], alpha=0.7)
        l, r = a.get_xlim()
        x_line = np.linspace(l, r, 20)
        y_line = (ort[1, 0] / ort[0, 0]) * x_line if ort[0, 0] != 0 else np.zeros_like(x_line)
        a.plot(x_line, y_line, c="orange", lw=2)

    denom = np.dot(ort.T, ort)
    proj_matrix = np.dot(ort, ort.T) / denom if denom > 0 else np.zeros((2, 2))
    for i in range(len(X_arr)):
        m = np.dot(proj_matrix, X_arr[i, :2].reshape(2, 1)).flatten()
        ax[1].plot([X_arr[i, 0], m[0]], [X_arr[i, 1], m[1]], c="green", alpha=0.5)
        ax[1].scatter(m[0], m[1], marker="x", c="k")

    fig.tight_layout()
    plt.show()


def plot_variance_explained(ve):
    """
    Plot individual and cumulative proportion of variance explained by principal components.
    """
    ve_arr = np.asarray(ve, dtype=float)
    coeffs = ve_arr / ve_arr.sum()
    cumulative = np.cumsum(coeffs)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    ax[0].bar(range(1, len(ve_arr) + 1), ve_arr, color="royalblue")
    ax[0].set_ylabel("Variance explained")
    ax[0].set_xlabel("Principal Component")

    ax[1].bar(range(1, len(ve_arr) + 1), cumulative, color="forestgreen")
    ax[1].set_ylabel("Cumulative variance explained")
    ax[1].set_xlabel("Principal Component")
    ax[1].set_ylim(0, 1.05)

    fig.tight_layout()
    plt.show()


def plot_pca_biplot(z1, z2, sc, comps, obs, features, colors):
    """
    Render PCA biplot with observation scores and feature loading vectors.
    """
    x, y = np.asarray(sc)[:, z1], np.asarray(sc)[:, z2]
    comps_arr = np.asarray(comps)

    fig = plt.figure(figsize=(10, 10))
    plt.xlabel("$z_{%d}$" % z1)
    plt.ylabel("$z_{%d}$" % z2)

    sx = (x.max() - x.min()) / 2
    sy = (y.max() - y.min()) / 2

    plt.scatter(x, y, alpha=0.6)
    for i in range(len(obs)):
        plt.text(x[i], y[i], str(obs[i]), ha="center", fontsize=11, alpha=0.8)

    vec = comps_arr[[z1, z2], :].T
    for i in range(len(vec)):
        plt.arrow(
            0, 0,
            vec[i, 0] * sx,
            vec[i, 1] * sy,
            ec=colors[i % len(colors)],
            head_width=0.08,
            head_length=0.08,
            fc=colors[i % len(colors)],
            lw=1.5
        )
        plt.text(
            vec[i, 0] * sx * 1.15,
            vec[i, 1] * sy * 1.15,
            features[i],
            color=colors[i % len(colors)],
            fontsize=12,
            fontweight="bold"
        )

    plt.grid(True)
    plt.show()


def compress_image_svd(img, comps=15, std_pca_fn=None):
    """
    Demonstrate low-rank image reconstruction via SVD/PCA across RGB channels.
    """
    img_arr = np.asarray(img, dtype=np.float32)
    channels = []
    colors = ["red", "green", "blue"]

    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(15, 4))
    for i, c in enumerate(colors):
        channel = img_arr[:, :, i]
        if std_pca_fn is not None:
            try:
                import torch
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

        ve_norm = ve_arr[:comps] / ve_arr.sum()
        ax[i].bar(range(1, comps + 1), ve_norm, color=c)
        ax[i].set_title(f"{c.capitalize()} Channel Variance")
        ch_min, ch_max = ch_recon.min(), ch_recon.max()
        ch_norm = (ch_recon - ch_min) / (ch_max - ch_min) if ch_max > ch_min else ch_recon
        channels.append(ch_norm)

    fig.tight_layout()
    plt.show()

    recon = np.stack(channels, axis=-1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].grid(False)
    axs[1].imshow(recon)
    axs[1].set_title(f"Reconstructed ({comps} components)")
    axs[1].grid(False)
    plt.show()
