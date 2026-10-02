"""
Self-contained helper utilities for the AI and Machine Learning course.
Provides standalone plotting and data helpers decoupled from 03_DL.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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
    plt.plot([pred, pred], [resp, est], 'bo--')
    plt.plot(pred, est, 'ro')
    plt.plot(l, f(l), 'r')
    plt.show()


def plot_binary(predictor, X, y):
    """
    Plot decision boundaries for 2D binary classification.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c='w')

    # Plot decision boundaries
    ax = plt.gca()
    y1, y2 = ax.get_ylim()
    x1, x2 = ax.get_xlim()
    xm, ym = np.meshgrid(
        np.arange(x1, x2, (x2 - x1) / 100),
        np.arange(y1, y2, (y2 - y1) / 100)
    )
    p = predictor(np.c_[xm.ravel(), ym.ravel()]).reshape(xm.shape)
    plt.scatter(xm, ym, c=p, cmap='coolwarm', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='w', s=100, linewidths=2)
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
    return {'and': fn_and, 'or': fn_or, 'xor': fn_xor}.get(fn_name)
