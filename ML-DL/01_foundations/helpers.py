"""
Self-contained helper utilities for Mathematical Foundations (ML-DL / 01_foundations).
Provides visualization, vector geometry, PCA/SVD decomposition, and probability routines.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch


# ---------------------------------------------------------------------------
# Probability & Regression Simulation (02_probability)
# ---------------------------------------------------------------------------

def corr(x, y, **kwargs):
    """Annotate seaborn pairgrid subplots with Pearson correlation coefficient."""
    ax = plt.gca()
    r, p = stats.pearsonr(x, y)
    ax.annotate(f"r = {r:.2f}\np-value = {p:.2f}", xy=(0.1, 0.3), size=14, xycoords=ax.transAxes)


def plot_bias_variance(X, t, degrees=5, repeats=1000, f=None, sigma_epsilon=1.0):
    """Demonstrate bias-variance decomposition across polynomial model complexities."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures

    if f is None:
        f = lambda x: (x**2) / 2 - 3 * x + 5 + np.sin(2 * x)

    N = len(X)
    train_squared_error = np.zeros((degrees, repeats))
    test_squared_error = np.zeros((degrees, repeats))
    y_pred_store = np.zeros((degrees, repeats, N))
    X_T = X.reshape(-1, 1)

    for d in range(degrees):
        for r in range(repeats):
            X_deg = PolynomialFeatures(degree=d + 1, include_bias=False).fit_transform(X_T)
            X_train, X_test, t_train, t_test = train_test_split(X_deg, t, test_size=round(0.8 * N))
            model = LinearRegression().fit(X_train, t_train)
            y_test = model.predict(X_test)
            train_squared_error[d, r] = np.mean((t_train - model.predict(X_train)) ** 2)
            test_squared_error[d, r] = np.mean((y_test - t_test) ** 2)
            y_pred_store[d, r, :] = model.predict(X_deg)

    bias_squared = (np.mean(y_pred_store, 1) - f(X)) ** 2
    var_y_pred = np.var(y_pred_store, 1)

    d_plt = range(1, degrees + 1)
    plt.figure(figsize=(10, 10))
    plt.xlim([1, degrees])
    plt.ylim([0, 3])
    plt.xticks(d_plt)
    plt.plot(d_plt, np.mean(test_squared_error, 1), 'r', linewidth=3)
    plt.plot(d_plt, np.mean(bias_squared, 1), 'g', linewidth=3)
    plt.plot(d_plt, np.mean(var_y_pred, 1), 'b', linewidth=3)
    plt.plot(d_plt, (sigma_epsilon ** 2) * np.ones_like(d_plt), 'y', linewidth=3)
    plt.plot(d_plt, np.mean(train_squared_error, 1), 'k', linewidth=3)
    plt.legend([
        r'test squared error - expected $\mathbb{E}[(t - y(x))^2]$',
        r'expected $(\mathrm{bias}[y(x)])^2$',
        r'expected $\mathrm{var}[y(x)]$',
        r'$\mathrm{var}[\epsilon] = \sigma^2$',
        'training squared error'
    ], loc='upper center', fontsize=12)
    ax = plt.gca()
    ax.axvspan(1, 2, alpha=0.3, color='gray')
    ax.axvspan(4, 5, alpha=0.3, color='gray')
    plt.text(1.2, 2.8, 'underfitting')
    plt.text(4.2, 2.8, 'overfitting')
    plt.show()


# ---------------------------------------------------------------------------
# Linear Algebra & Vector Geometry (04_algebra)
# ---------------------------------------------------------------------------

def plot2d(tensor, title=None):
    """Plot 2D vectors emanating from the origin using quiver."""
    plt.figure(figsize=(5, 5))
    if hasattr(tensor, "detach"):
        t_np = tensor.detach().cpu().numpy()
    else:
        t_np = np.asarray(tensor)
    o = np.zeros_like(t_np)
    plt.quiver(*o, *t_np, angles="xy", scale_units="xy", scale=1, color=["r", "g", "b", "m"])
    mx = float(np.max(np.abs(t_np))) + 0.5
    plt.xlim(-mx, mx)
    plt.ylim(-mx, mx)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()


def std_full(T):
    means = T.mean(dim=0, keepdim=True)
    stds = T.std(dim=0, keepdim=True)
    return ((T - means) / stds), means, stds


def standardize(T):
    return std_full(T)[0]


def std_inverse(T, means, stds):
    k = T.size()[1]
    return T * stds[:, :k] + means[:, :k]


def normalize(T):
    if hasattr(T, "clone"):
        Tv = T.view(T.size(0), -1).clone()
    else:
        Tv = np.copy(T).reshape(T.shape[0], -1)
    Tv -= Tv.min()
    Tv /= Tv.max()
    return Tv


def to_coefficients(T):
    if hasattr(T, "sum"):
        return T / T.sum(0, keepdim=True)[0]
    return T / np.sum(T, axis=0, keepdims=True)[0]


def scale_vector(t_vec, alpha=1):
    """Scale a vector and render it with plot2d."""
    if alpha != 0:
        plot2d(torch.cat((t_vec, t_vec * alpha), dim=1), title=f"Scale factor: alpha={alpha}")
    else:
        plot2d(t_vec, title="Scale factor: alpha=0")


def inspect_angle(y1=0.0, y2=1.0):
    """Inspect angle and dot product between (1, 0) and (y1, y2)."""
    x_base = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    y_comp = torch.tensor([[float(y1)], [float(y2)]], dtype=torch.float32)
    cos_val = (torch.matmul(x_base.t(), y_comp) / (torch.norm(x_base) * torch.norm(y_comp))).clamp(-1.0, 1.0).item()
    deg = math.degrees(math.acos(cos_val))
    print(f"Inner product: {torch.matmul(x_base.t(), y_comp).item():.3f} | Cosine: {cos_val:.3f} | Angle: {deg:.1f} deg")
    plot2d(torch.cat((x_base, y_comp), dim=1), title=f"Angle: {deg:.1f} deg")


def rotate_vector(alpha=90):
    """Rotate base vector (1, 0) by angle alpha degrees."""
    rad = math.radians(alpha)
    rot_mat = torch.tensor([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=torch.float32)
    v_init = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    v_rot = torch.matmul(rot_mat, v_init)
    print(f"Angle {alpha} deg: rotated vector = ({v_rot[0,0].item():.3f}, {v_rot[1,0].item():.3f})")
    plot2d(torch.cat((v_rot, v_init), dim=1), title=f"Rotation by {alpha} deg")


def plot_scores(X_us, us_columns, x=0, y=0):
    """Plot 2D PCA orthogonal projection and orientation lines."""
    ort = torch.tensor([[float(x)], [float(y)]], dtype=torch.float32)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    ax[0].set_ylabel(us_columns[1])
    X_plot = X_us.numpy() if hasattr(X_us, "numpy") else np.asarray(X_us)

    for a in ax:
        a.set_xlabel(us_columns[0])
        a.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.7)
        l, r = a.get_xlim()
        if abs(ort[0].item()) > 1e-6:
            x_vals = np.linspace(l, r, 20)
            y_vals = (ort[1].item() / ort[0].item()) * x_vals
            a.plot(x_vals, y_vals, c='orange', lw=2)

    denom = torch.matmul(ort.t(), ort).item()
    if abs(denom) > 1e-6:
        proj_matrix = torch.matmul(ort, ort.t()) / denom
        X_sub = X_us[:, :2] if hasattr(X_us, "device") else torch.tensor(X_us[:, :2], dtype=torch.float32)
        projected = torch.matmul(X_sub, proj_matrix).numpy()
        for i in range(len(X_plot)):
            ax[1].plot([X_plot[i, 0], projected[i, 0]], [X_plot[i, 1], projected[i, 1]], c='green', alpha=0.4)
        ax[1].scatter(projected[:, 0], projected[:, 1], marker='x', c='k', s=25)

    fig.tight_layout()
    plt.show()


def biplot(z1, z2, sc, comps, obs, features, colors):
    """Render PCA biplot with observation scores and loading vectors."""
    x = sc[:, z1].numpy() if hasattr(sc, "numpy") else np.asarray(sc)[:, z1]
    y = sc[:, z2].numpy() if hasattr(sc, "numpy") else np.asarray(sc)[:, z2]
    fig = plt.figure(figsize=(9, 8))
    plt.xlabel(f"$z_{{{z1+1}}}$")
    plt.ylabel(f"$z_{{{z2+1}}}$")

    sx = float((x.max() - x.min()) / 2)
    sy = float((y.max() - y.min()) / 2)

    plt.scatter(x, y, alpha=0.6)
    for i in range(len(obs)):
        plt.text(x[i], y[i], str(obs[i]), ha='center', fontsize=9, alpha=0.8)

    comps_mat = comps[[z1, z2], :].numpy().T if hasattr(comps, "numpy") else np.asarray(comps)[[z1, z2], :].T
    for i in range(len(comps_mat)):
        plt.arrow(0, 0, comps_mat[i, 0] * sx, comps_mat[i, 1] * sy, ec=colors[i % len(colors)],
                  head_width=0.1, head_length=0.1, fc=colors[i % len(colors)], lw=2)
        plt.text(comps_mat[i, 0] * sx * 1.15, comps_mat[i, 1] * sy * 1.15, features[i],
                 color=colors[i % len(colors)], fontsize=12, weight='bold')

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compress_image_svd(img, comps=15, std_pca_fn=None):
    """RGB channel low-rank image reconstruction via SVD/PCA."""
    img_arr = np.asarray(img, dtype=np.float32)
    channels = []
    ch_colors = ['red', 'green', 'blue']
    ch_names = ['Red Channel', 'Green Channel', 'Blue Channel']

    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(12, 3.5))
    for i in range(3):
        channel = img_arr[:, :, i]
        if std_pca_fn is not None:
            ch_in = torch.tensor(channel, dtype=torch.float32)
            sc, cp, ve, _ = std_pca_fn(ch_in, k=comps)
            ch_recon = torch.matmul(sc, cp).numpy()
            ve_arr = ve.numpy() if hasattr(ve, "numpy") else np.asarray(ve)
        else:
            U, S, Vt = np.linalg.svd(channel, full_matrices=False)
            ch_recon = np.dot(U[:, :comps] * S[:comps], Vt[:comps, :])
            ve_arr = S ** 2

        ve_norm = ve_arr[:comps] / ve_arr.sum()
        ax[i].bar(range(1, comps + 1), ve_norm, color=ch_colors[i])
        ax[i].set_title(ch_names[i])
        ax[i].set_xlabel("Component")

        ch_min, ch_max = ch_recon.min(), ch_recon.max()
        ch_norm = (ch_recon - ch_min) / (ch_max - ch_min) if ch_max > ch_min else ch_recon
        channels.append(ch_norm)

    fig.tight_layout()
    plt.show()

    recon = np.stack(channels, axis=-1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].grid(False)
    axs[1].imshow(recon)
    axs[1].set_title(f"Reconstruction ({comps} components)")
    axs[1].grid(False)
    plt.show()
