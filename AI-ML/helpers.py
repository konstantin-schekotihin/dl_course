"""
Self-contained helper utilities for the AI and Machine Learning course.
Provides standalone plotting and data helpers decoupled from 03_DL.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import sklearn.metrics as metrics
import torch


def annotate(x, y, **kws):
    """Annotate seaborn pairgrid subplots with Pearson correlation coefficient."""
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}, p = {:.3f} ".format(r, p),
                xy=(.1, 1), xycoords=ax.transAxes)


def plot_residuals(f, pred, resp, show=False):
    """Plot regression line and residuals between predictions and responses."""
    est = f(pred.reshape(-1, 1))
    if show:
        rss = np.sum(np.power(resp - est, 2))
        tss = np.sum(np.power(resp - np.average(resp), 2))
        print("TSS = {:.3f} - total sum of squares - squared error of an average predictor".format(tss))
        print("RSS = {:.3f} - residual sum of squares".format(rss))
        print("ESS = TSS - RSS =  {:.3f} - {:.3f} = {:.3f} - explained sum of squares".format(tss, rss, tss - rss))
        print("MSE = {:.3f} - mean squared error".format(rss / est.shape[0]))
        print("MAE = {:.3f} - mean absolute error".format(np.sum(np.abs(resp - est)) / est.shape[0]))

    l = np.linspace(0, 300, num=1000).reshape(-1, 1)
    plt.figure(figsize=(10, 10))
    plt.plot([pred, pred], [resp, est], "bo--")
    plt.plot(pred, est, "ro")
    plt.plot(l, f(l), "r")
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


def rescale(X, min_val=-1, max_val=1):
    """
    Rescale tensor or array into [min_val, max_val].
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std * (max_val - min_val) + min_val


def fn_and(X):
    """Logical AND for 2D inputs."""
    return torch.logical_and(X[:, 0], X[:, 1]).float()


def fn_or(X):
    """Logical OR for 2D inputs."""
    return torch.logical_or(X[:, 0], X[:, 1]).float()


def fn_xor(X):
    """Logical XOR for 2D inputs."""
    return torch.logical_xor(X[:, 0], X[:, 1]).float()


def switch_fn(fn_name):
    """Return logic function by name."""
    return {"and": fn_and, "or": fn_or, "xor": fn_xor}.get(fn_name)


def plot_knn_results(classifier, train_X, train_y, test_X, test_y, target_names=None, feature_cols=None):
    """
    Plot dual-subplot evaluation for kNN:
    - Left: Confusion matrix heatmap.
    - Right: 2D scatter plot showing training data and test classification errors (faults).
    """
    pred_y = classifier.predict(test_X)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    acc = np.sum(pred_y == test_y) / float(len(test_y))
    print("Accuracy: {:.4f}".format(acc))
    print("Correctly classified:", np.where(pred_y == test_y)[0])
    print("Samples incorrectly classified:", np.where(pred_y != test_y)[0])

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

    print("Colors indicate correct classes and markers an error")
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
