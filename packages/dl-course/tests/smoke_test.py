#!/usr/bin/env python3
"""
Smoke test for the dl-course helper package.

Calls every public helper once with tiny synthetic inputs (Agg backend, no windows) and
fails if any call raises, or if a public helper has no test case below. It runs in the
publish workflow before every release, and locally before committing a helper change.

Usage (from the repository root):
    uv run python packages/dl-course/tests/smoke_test.py
"""

import importlib
import inspect
import os
import sys
import traceback

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

MODULES = ["shared", "aiml", "foundations", "deep_learning"]
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # tests -> dl-course -> packages -> repo

plt.show = lambda *args, **kwargs: None  # plots are rendered off-screen and discarded

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
x = torch.linspace(0.0, 1.0, 20)
t = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn(20)
x_dense = torch.linspace(0.0, 1.0, 100)
t_dense = torch.sin(2 * torch.pi * x_dense)
w2, w4, w10 = torch.zeros(2), torch.zeros(4), torch.zeros(10)

X2 = rng.normal(size=(40, 2))
y2 = (X2[:, 0] + X2[:, 1] > 0).astype(int)
X4 = rng.normal(size=(40, 4))
labels = np.where(y2 == 1, "a", "b")
logreg = LogisticRegression().fit(X2, y2)
svc = SVC(kernel="linear").fit(X2, y2)

df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["u", "v", "w"])
iris = pd.DataFrame({
    "sepal_length": rng.uniform(4, 8, 30),
    "petal_length": rng.uniform(1, 7, 30),
    "species": ["setosa", "virginica", "versicolor"] * 10,
})
img = rng.uniform(size=(16, 16, 3)).astype(np.float32)
img2d = img[:, :, 0]
sc, comps = rng.normal(size=(10, 3)), rng.normal(size=(3, 3))
obs, feats, cols = [f"o{i}" for i in range(10)], ["f1", "f2", "f3"], ["r", "g", "b"]


def first_local_image():
    folder = os.path.join(REPO_ROOT, "shared", "images")
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    return names[0]


# (module, public name, call) -- one case per distinct function; aliases are covered by their target.
CASES = [
    # shared
    ("shared", "print_math", lambda hp: hp.print_math(r"$x$")),
    ("shared", "get_url", lambda hp: hp.get_url("Advertising.csv")),
    ("shared", "get_file", lambda hp: hp.get_file("Advertising.csv")),
    ("shared", "get_data_path", lambda hp: hp.get_data_path("Advertising.csv")),
    ("shared", "setup_theme", lambda hp: hp.setup_theme()),
    ("shared", "plot_residuals", lambda hp: hp.plot_residuals(lambda X: 2 * X, x, t)),
    ("shared", "plot_binary", lambda hp: hp.plot_binary(logreg.predict, X2, y2)),
    ("shared", "plot_correlation_matrix", lambda hp: hp.plot_correlation_matrix(df)),
    ("shared", "plot_pairgrid", lambda hp: hp.plot_pairgrid(df)),
    ("shared", "plot_correlation_matrix_and_pairgrid", lambda hp: hp.plot_correlation_matrix_and_pairgrid(df)),
    ("shared", "annotate", lambda hp: (
        hp.annotate(img2d[:2, :2], plt.gca()),
        hp.annotate(plt.gca(), "badge"),
        hp.annotate(df["u"].to_numpy(), df["v"].to_numpy()),
    )),
    ("shared", "load_image", lambda hp: hp.load_image(first_local_image(), as_tensor=True)),
    ("shared", "plot_image", lambda hp: hp.plot_image(img)),
    ("shared", "compress_image_svd", lambda hp: hp.compress_image_svd(img, comps=3)),
    ("shared", "to_numpy", lambda hp: hp.to_numpy(x)),
    ("shared", "to_scalar", lambda hp: hp.to_scalar(x[0])),
    ("shared", "ensure_2d", lambda hp: hp.ensure_2d(x)),
    ("shared", "eval_predictor", lambda hp: hp.eval_predictor(logreg, X2)),
    ("shared", "get_boundaries", lambda hp: hp.get_boundaries(X2)),
    ("shared", "create_2d_grid", lambda hp: hp.create_2d_grid(X2, resolution=20)),
    # aiml
    ("aiml", "plot_residuals", lambda hp: hp.plot_residuals(lambda X: 2 * X, X2[:, 0], X2[:, 1])),
    ("aiml", "plot_linear_classification_fit", lambda hp: hp.plot_linear_classification_fit(X2[:, 0], y2, lambda l: 0.5 * l)),
    ("aiml", "plot_logistic_response_curve", lambda hp: hp.plot_logistic_response_curve(x, t, w_0=1.0, w_1=2.0)),
    ("aiml", "plot_logistic_curve", lambda hp: hp.plot_logistic_curve(x, t, 1.0, 2.0)),
    ("aiml", "plot_positive_vs_log", lambda hp: hp.plot_positive_vs_log(x_dense + 1, x_dense + 1, np.log(x_dense + 1))),
    ("aiml", "plot_curves", lambda hp: hp.plot_curves([1, 2, 3], {"train": [3.0, 2.0, 1.0], "test": [3.0, 2.5, 2.8]}, mark_x=2)),
    ("aiml", "plot_fitted_curves", lambda hp: hp.plot_fitted_curves(x, t, x_dense, {"fit": torch.sin(2 * torch.pi * x_dense)})),
    ("aiml", "plot_confounding", lambda hp: hp.plot_confounding([1, 2, 3], {"A": [0.1, 0.2, 0.3], "B": [0.2, 0.3, 0.4]}, {"A": 0.2, "B": 0.3}, {"A": [1, 2, 3], "B": [2, 3, 4]}, "x", "rate", "x")),
    ("aiml", "plot_roc_curves", lambda hp: hp.plot_roc_curves({"model": ([0.0, 0.2, 1.0], [0.0, 0.7, 1.0])})),
    ("aiml", "plot_residual_panels", lambda hp: hp.plot_residual_panels({"A": (x, t), "B": (t, x)}, band=1.0, n_labels=2)),
    ("aiml", "get_boundaries", lambda hp: hp.get_boundaries(X2)),
    ("aiml", "plot_knn_neighborhood_2d", lambda hp: hp.plot_knn_neighborhood_2d(X2, labels, show_neighborhood=True)),
    ("aiml", "plot_knn_intuition", lambda hp: hp.plot_knn_intuition(iris, show=True)),
    ("aiml", "plot_classes_2d", lambda hp: hp.plot_classes_2d(X2, y2)),
    ("aiml", "plot_class_counts", lambda hp: hp.plot_class_counts({"all": iris, "half": iris.iloc[::2]}, x="species", hue="species")),
    ("aiml", "plot_knn_classification_results", lambda hp: hp.plot_knn_classification_results(X4, labels, X4, labels, labels[::-1])),
    ("aiml", "plot_knn_results", lambda hp: hp.plot_knn_results(X4, labels, X4, labels, labels[::-1])),
    ("aiml", "plot_confusion_matrix", lambda hp: hp.plot_confusion_matrix(y2, y2[::-1])),
    ("aiml", "plot_decision_boundary", lambda hp: hp.plot_decision_boundary(logreg, X2, y2)),
    ("aiml", "plot_decision_boundaries_comparison", lambda hp: hp.plot_decision_boundaries_comparison({"A": logreg, "B": logreg}, X2, y2)),
    ("aiml", "plot_svm_margins", lambda hp: hp.plot_svm_margins(svc)),
    ("aiml", "plot_svm_separating_plane", lambda hp: hp.plot_svm_separating_plane(X2, y2, show_margin=True)),
    ("aiml", "plot_feature_importances", lambda hp: hp.plot_feature_importances(["f1", "f2"], {"RF": [0.6, 0.4]})),
    ("aiml", "plot_pca_scores", lambda hp: hp.plot_pca_scores(X2, np.array([1.0, 0.5]))),
    ("aiml", "plot_variance_explained", lambda hp: hp.plot_variance_explained([3.0, 2.0, 1.0])),
    ("aiml", "plot_pca_biplot", lambda hp: hp.plot_pca_biplot(0, 1, sc, comps, obs, feats, cols)),
    ("aiml", "plot_loading_heatmap", lambda hp: hp.plot_loading_heatmap(comps, x_labels=feats)),
    # foundations
    ("foundations", "plot_gaussian_vs_uniform", lambda hp: hp.plot_gaussian_vs_uniform(rng.normal(0, 20, 300), rng.normal(0, 10, 300), rng.uniform(-100, 100, 300))),
    ("foundations", "plot_regression_dataset", lambda hp: hp.plot_regression_dataset(x.numpy(), t.numpy(), lambda v: np.sin(2 * np.pi * v))),
    ("foundations", "plot_gmm_samples", lambda hp: hp.plot_gmm_samples(X2)),
    ("foundations", "render_bias_variance_plot", lambda hp: hp.render_bias_variance_plot([1, 2, 3], [3, 2, 1], [3, 2, 3], [2, 1, 0.5], [0.1, 0.5, 1])),
    ("foundations", "plot_density_comparison", lambda hp: hp.plot_density_comparison(x_dense.numpy(), {"p": t_dense.numpy()})),
    ("foundations", "plot_joint_distribution_heatmap", lambda hp: hp.plot_joint_distribution_heatmap(np.full((2, 3), 1 / 6))),
    ("foundations", "plot2d", lambda hp: hp.plot2d(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))),
    ("foundations", "scale_vector", lambda hp: hp.scale_vector(torch.tensor([[1.0], [2.0]]), 2.0)),
    ("foundations", "inspect_angle", lambda hp: hp.inspect_angle(1.0, 1.0)),
    ("foundations", "rotate_vector", lambda hp: hp.rotate_vector(45.0)),
    ("foundations", "plot_out", lambda hp: hp.plot_out([1.0, 0.1, 0.01])),
    ("foundations", "plot_scores", lambda hp: hp.plot_scores(torch.tensor(X2[:10], dtype=torch.float32), ["a", "b"], 1.0, 0.5)),
    ("foundations", "biplot", lambda hp: hp.biplot(0, 1, sc, comps, obs, feats, cols)),
    ("foundations", "plot_pca_vectors", lambda hp: hp.plot_pca_vectors(torch.tensor(X2), ["a", "b"], torch.eye(2))),
    ("foundations", "plot_loadings_heatmap", lambda hp: hp.plot_loadings_heatmap(torch.tensor(comps), feats)),
    ("foundations", "plot_explained_variance", lambda hp: hp.plot_explained_variance(torch.tensor([3.0, 2.0, 1.0]))),
    ("foundations", "plot_iris_pairplot", lambda hp: hp.plot_iris_pairplot(iris)),
    ("foundations", "plot_function_and_derivative", lambda hp: hp.plot_function_and_derivative(x_dense, t_dense, t_dense)),
    ("foundations", "plot_multi_curve_comparison", lambda hp: hp.plot_multi_curve_comparison(x_dense, {"sin": t_dense})),
    ("foundations", "plot_curve_with_marker", lambda hp: hp.plot_curve_with_marker(x_dense, t_dense, 0.5, 0.0, vline_at=0.5)),
    ("foundations", "plot_categorical_probabilities", lambda hp: hp.plot_categorical_probabilities(["a", "b"], {"T=1": [0.3, 0.7]})),
    ("foundations", "plot_classifier_diagnostics", lambda hp: hp.plot_classifier_diagnostics(np.array([[3, 1], [1, 3]]), rng.uniform(size=20))),
    # deep_learning
    ("deep_learning", "plot_decision_boundary_2d", lambda hp: hp.plot_decision_boundary_2d(logreg.predict, X2, y2)),
    ("deep_learning", "plot_synthetic_regression_data", lambda hp: hp.plot_synthetic_regression_data(x_dense, t_dense, x, t)),
    ("deep_learning", "plot_regression_fit", lambda hp: hp.plot_regression_fit(x, t, x_dense, t_dense, y_true=t_dense, residuals=True)),
    ("deep_learning", "plot_polynomial_fits", lambda hp: hp.plot_polynomial_fits(x_dense, t_dense, x, t, {1: w2, 3: w4}, [1, 3])),
    ("deep_learning", "plot_polynomial_extrapolation", lambda hp: hp.plot_polynomial_extrapolation(x, t, w4, x_range=(-0.5, 1.5))),
    ("deep_learning", "plot_dataset_size_remedy", lambda hp: hp.plot_dataset_size_remedy(x_dense, t_dense, [(20, x, t, t_dense), (20, x, t, t_dense)])),
    ("deep_learning", "plot_regularized_fits", lambda hp: hp.plot_regularized_fits(x_dense, t_dense, x, t, {-18: w10, 0: w10}, [-18, 0])),
    ("deep_learning", "plot_basis_functions", lambda hp: hp.plot_basis_functions(x_dense, [0.2, 0.5])),
    ("deep_learning", "plot_bias_variance_curve", lambda hp: hp.plot_bias_variance_curve([-2, 0, 2], [0.1, 0.05, 0.01], [0.01, 0.05, 0.1], [0.11, 0.1, 0.11])),
    ("deep_learning", "plot_ensemble_fits", lambda hp: hp.plot_ensemble_fits(x_dense, t_dense, {k: ([t_dense] * 3, t_dense) for k in (2.0, -0.5, -4.0)})),
    ("deep_learning", "plot_l1_l2_sparsity_geometry", lambda hp: hp.plot_l1_l2_sparsity_geometry()),
    ("deep_learning", "plot_l1_l2_outlier_comparison", lambda hp: hp.plot_l1_l2_outlier_comparison(x, t, x_dense, t_dense, t_dense)),
    ("deep_learning", "plot_streaming_adaline_convergence", lambda hp: hp.plot_streaming_adaline_convergence(
        x, t, x_dense, torch.stack([torch.ones(100), x_dense], dim=1), [(0, w2), (1, w2), (3, w2)], w2)),
    ("deep_learning", "plot_loss_history", lambda hp: hp.plot_loss_history([1.0, 0.5, 0.1])),
    ("deep_learning", "plot_out", lambda hp: hp.plot_out([1.0, 0.5, 0.1])),
    ("deep_learning", "g_idea", lambda hp: hp.g_idea(lambda v: v ** 2, lambda v: 2 * v)),
    ("deep_learning", "plot_trace", lambda hp: hp.plot_trace(lambda v: v ** 2, [1.0, 0.5, 0.25])),
    ("deep_learning", "plot_grad", lambda hp: hp.plot_grad(lambda a, b: a ** 2 + b ** 2, 1.0, 1.0, 0.5, 0.5)),
    ("deep_learning", "plot_activation_curves", lambda hp: hp.plot_activation_curves("gelu")),
    ("deep_learning", "plot_sobel_filtered", lambda hp: hp.plot_sobel_filtered(img2d, img2d)),
    ("deep_learning", "plot_time_series", lambda hp: hp.plot_time_series(list(range(10)), list(range(10)), label="s")),
    ("deep_learning", "plot_attention_heatmap", lambda hp: hp.plot_attention_heatmap(np.full((3, 3), 1 / 3), ["a", "b", "c"])),
    ("deep_learning", "plot_graph", lambda hp: hp.plot_graph(nx.karate_club_graph())),
    ("deep_learning", "plot_embedding", lambda hp: hp.plot_embedding(rng.normal(size=(10, 2)), list(range(10)), epoch=1, loss=0.5)),
]


def public_functions(module):
    """Distinct helper functions reachable through a module's public names."""
    found = {}
    for name, obj in vars(module).items():
        if name.startswith("_") or not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            continue
        func = getattr(obj, "__func__", obj)
        if func.__module__.startswith("dl_course"):
            found.setdefault(func, name)
    return found


def main() -> int:
    os.chdir(REPO_ROOT)
    import dl_course
    print(f"dl-course version: {dl_course.__version__}")
    modules = {name: importlib.import_module(f"dl_course.{name}") for name in MODULES}
    failures, covered = [], set()

    for module_name, attr, call in CASES:
        hp = modules[module_name]
        covered.add(getattr(getattr(hp, attr), "__func__", getattr(hp, attr)))
        try:
            call(hp)
            print(f"  ok    {module_name}.{attr}")
        except Exception:
            failures.append(f"{module_name}.{attr}")
            print(f"  FAIL  {module_name}.{attr}\n{traceback.format_exc()}")
        finally:
            plt.close("all")

    uncovered = sorted(
        f"{name}.{alias} ({func.__qualname__})"
        for name, module in modules.items()
        for func, alias in public_functions(module).items()
        if func not in covered
    )
    for item in uncovered:
        print(f"  MISSING CASE  {item}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} cases passed, {len(uncovered)} public helpers without a case")
    return 1 if failures or uncovered else 0


if __name__ == "__main__":
    sys.exit(main())
