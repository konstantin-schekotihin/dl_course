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
from pathlib import Path

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
    ("shared", "format_number", lambda hp: (hp.format_number(float("inf")), hp.format_number(0.5, digits=2), hp.format_number(1.5e-8, 1, scientific=True))),
    ("shared", "format_vector", lambda hp: (hp.format_vector(np.array([0.5, -0.0001, 2.0]), digits=2), hp.format_vector([3.0, float("inf")], transpose=False))),
    ("shared", "get_url", lambda hp: hp.get_url("Advertising.csv")),
    ("shared", "get_file", lambda hp: hp.get_file("Advertising.csv")),
    ("shared", "get_data_path", lambda hp: hp.get_data_path("Advertising.csv")),
    ("shared", "download", lambda hp: os.remove(hp.download(Path(hp.get_data_path("Advertising.csv")).as_uri(),
                                                            "smoke_download.csv"))),
    ("shared", "setup_theme", lambda hp: hp.setup_theme()),
    ("shared", "plot_residuals", lambda hp: hp.plot_residuals(lambda X: 2 * X, x, t)),
    ("shared", "plot_binary", lambda hp: hp.plot_binary(logreg.predict, X2, y2)),
    ("shared", "plot_correlation_matrix", lambda hp: hp.plot_correlation_matrix(df)),
    ("shared", "plot_pairgrid", lambda hp: hp.plot_pairgrid(df)),
    ("shared", "plot_decision_surface", lambda hp: (
        hp.plot_decision_surface(X2, y2, *hp.create_2d_grid(X2, resolution=20)[:2],
                                 values=(hp.create_2d_grid(X2, resolution=20)[0] > 0).astype(int),
                                 boundaries={"b": [0.0, 1.0, 1.0]}, highlight=[0]),
        hp.plot_decision_surface(X2, y2, *hp.create_2d_grid(X2, resolution=20)[:2],
                                 values=np.full((20, 20), 0.5) + 0.01 * rng.normal(size=(20, 20)), levels=[0.5]))),
    ("shared", "plot_confusion_matrix", lambda hp: hp.plot_confusion_matrix(np.array([[5, 1], [2, 4]]), labels=["neg", "pos"])),
    ("shared", "plot_roc_curves", lambda hp: hp.plot_roc_curves({"model": ([0.0, 0.2, 1.0], [0.0, 0.7, 1.0])}, points={"tau": (0.2, 0.7)})),
    ("shared", "annotate", lambda hp: (
        hp.annotate(img2d[:2, :2], plt.gca()),
        hp.annotate(plt.gca(), "badge"),
        hp.annotate(df["u"].to_numpy(), df["v"].to_numpy()),
    )),
    ("shared", "load_image", lambda hp: hp.load_image(first_local_image(), as_tensor=True)),
    ("shared", "plot_image", lambda hp: hp.plot_image(img)),
    ("shared", "plot_image_grid", lambda hp: hp.plot_image_grid(rng.uniform(size=(5, 8, 8)), n_cols=3)),
    ("shared", "plot_image_grid", lambda hp: hp.plot_image_grid([rng.normal(size=(3, 3)), rng.uniform(size=(3, 3))],
        signed=[True, False], annotate=True, overlays=[rng.uniform(size=(3, 3))] * 2)),
    ("shared", "plot_prediction", lambda hp: hp.plot_prediction(rng.uniform(size=(8, 8)), np.full(3, 1 / 3),
        ["a", "b", "c"], true_class=1, title="t")),
    ("deep_learning", "plot_boxes", lambda hp: hp.plot_boxes(rng.uniform(size=(20, 20)),
        {"truth": [[2, 2, 10, 10]], "pred": [[3, 3, 11, 12], [1, 1, 5, 5]]}, scores={"pred": [0.9, 0.4]})),
    ("shared", "compress_image_svd", lambda hp: hp.compress_image_svd(img, comps=3)),
    ("shared", "plot_function_panels", lambda hp: hp.plot_function_panels(x_dense, [{"f": t_dense, "g": x_dense}, {"f'": t_dense}], titles=["a", "b"], ylims=[None, (-1, 1)], points=[(0.5, 0.0), None], dashed=["g"])),
    ("shared", "plot_projection", lambda hp: hp.plot_projection(X2, np.outer(X2 @ [0.6, 0.8], [0.6, 0.8]), [0.6, 0.8], xlabel="a", ylabel="b")),
    ("shared", "plot_explained_variance", lambda hp: hp.plot_explained_variance([0.5, 0.3, 0.2], [0.5, 0.8, 1.0])),
    ("shared", "plot_directions", lambda hp: hp.plot_directions(X2, {"u": [1.0, 0.5]})),
    ("shared", "plot_loading_heatmap", lambda hp: hp.plot_loading_heatmap(comps, x_labels=feats)),
    ("shared", "plot_pca_biplot", lambda hp: (
        hp.plot_pca_biplot(0, 1, sc, comps, obs, feats, cols),
        hp.plot_pca_biplot(0, 1, sc, comps, None, feats, cols, groups=["a", "b"] * 5),
    )),
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
    ("aiml", "plot_positive_vs_log", lambda hp: hp.plot_positive_vs_log(x_dense + 1, x_dense + 1, np.log(x_dense + 1))),
    ("shared", "plot_curves", lambda hp: hp.plot_curves([1, 2, 3], {"train": [3.0, 2.0, 1.0], "test": [3.0, 2.5, 2.8]}, mark_x=2)),
    ("shared", "plot_fitted_curves", lambda hp: hp.plot_fitted_curves(x, t, x_dense, {"fit": torch.sin(2 * torch.pi * x_dense)}, points={"p": (0.5, 0.0)}, ylim=(-2, 2))),
    ("aiml", "plot_confounding", lambda hp: hp.plot_confounding([1, 2, 3], {"A": [0.1, 0.2, 0.3], "B": [0.2, 0.3, 0.4]}, {"A": 0.2, "B": 0.3}, {"A": [1, 2, 3], "B": [2, 3, 4]}, "x", "rate", "x")),
    ("aiml", "plot_residual_panels", lambda hp: hp.plot_residual_panels({"A": (x, t), "B": (t, x)}, band=1.0, n_labels=2)),
    ("aiml", "get_boundaries", lambda hp: hp.get_boundaries(X2)),
    ("aiml", "plot_knn_neighborhood_2d", lambda hp: hp.plot_knn_neighborhood_2d(X2, labels, show_neighborhood=True)),
    ("aiml", "plot_classes_2d", lambda hp: hp.plot_classes_2d(X2, y2)),
    ("aiml", "plot_class_counts", lambda hp: hp.plot_class_counts({"all": iris, "half": iris.iloc[::2]}, x="species", hue="species")),
    ("aiml", "plot_knn_classification_results", lambda hp: hp.plot_knn_classification_results(X4, labels, X4, labels, labels[::-1])),
    ("aiml", "plot_decision_boundary", lambda hp: hp.plot_decision_boundary(logreg, X2, y2)),
    ("aiml", "plot_decision_boundaries_comparison", lambda hp: hp.plot_decision_boundaries_comparison({"A": logreg, "B": logreg}, X2, y2)),
    ("aiml", "plot_svm_panels", lambda hp: hp.plot_svm_panels({"A": (X2, y2, *np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20)), np.zeros((20, 20)) + np.linspace(-3, 3, 20), X2[:3])})),
    ("aiml", "plot_svm_separating_plane", lambda hp: hp.plot_svm_separating_plane(X2, y2)),
    ("aiml", "plot_feature_importances", lambda hp: hp.plot_feature_importances(["f1", "f2"], {"RF": [0.6, 0.4]})),
    ("aiml", "plot_pca_scores", lambda hp: hp.plot_pca_scores(X2, np.array([1.0, 0.5]))),
    ("aiml", "plot_variance_explained", lambda hp: hp.plot_variance_explained([3.0, 2.0, 1.0])),
    ("aiml", "plot_dendrograms", lambda hp: hp.plot_dendrograms({"complete": __import__("scipy.cluster.hierarchy", fromlist=["linkage"]).linkage(X2[:8], "complete")}, cut_heights={"complete": 1.0})),
    # foundations
    ("foundations", "plot_sample_histograms", lambda hp: hp.plot_sample_histograms({"a": rng.normal(0, 1, 300), "b": rng.uniform(-2, 2, 300)})),
    ("foundations", "plot_gmm_samples", lambda hp: hp.plot_gmm_samples(X2)),
    ("foundations", "plot_joint_distribution_heatmap", lambda hp: hp.plot_joint_distribution_heatmap(np.full((2, 3), 1 / 6))),
    ("foundations", "plot2d", lambda hp: hp.plot2d(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))),
    ("foundations", "plot_convergence", lambda hp: hp.plot_convergence([1.0, 0.1, 0.01])),
    ("foundations", "plot_iris_pairplot", lambda hp: hp.plot_iris_pairplot(iris)),
    ("foundations", "plot_curve_with_marker", lambda hp: hp.plot_curve_with_marker(x_dense, t_dense, 0.5, 0.0, vline_at=0.5, hline_at=0.0)),
    ("foundations", "plot_categorical_probabilities", lambda hp: hp.plot_categorical_probabilities(["a", "b"], {"T=1": [0.3, 0.7]}, ylim=(0, 1))),
    # deep_learning
    ("deep_learning", "plot_descent_1d", lambda hp: hp.plot_descent_1d(x, x ** 2, [1.0, 0.5, 0.25], [1.0, 0.25, 0.0625])),
    ("deep_learning", "plot_contour_trajectories", lambda hp: hp.plot_contour_trajectories(
        *np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20)),
        np.add.outer(np.linspace(-2, 2, 20) ** 2, np.linspace(-2, 2, 20) ** 2),
        {"a": np.array([[1.0, 1.0], [0.5, 0.2], [0.0, 0.0]])}, directions={"u": [1.0, 0.0]},
        curves={"c": np.array([[1.0, 0.0], [0.0, 1.0]])}, points={"p": (0.5, 0.5)}, equal_aspect=True,
        minimum_label="m", legend_loc="lower right")),
    ("deep_learning", "plot_computational_graph", lambda hp: hp.plot_computational_graph(
        {"a": (0.0, 0.0, "$a$"), "b": (2.0, 0.0, "$b$")}, [("a", "b")], values={"a": "1"}, adjoints={"b": "2"},
        edge_labels={("a", "b"): "x"})),
    ("deep_learning", "plot_hyperplane_geometry", lambda hp: hp.plot_hyperplane_geometry([2.0, 1.0], -2.0, [2.0, 2.0], [0.4, 1.2])),
    ("deep_learning", "plot_reject_option", lambda hp: hp.plot_reject_option(x, {"a": x, "b": 1 - x}, 0.7, (x > 0.3) & (x < 0.7))),
    ("deep_learning", "plot_synthetic_regression_data", lambda hp: hp.plot_synthetic_regression_data(x_dense, t_dense, x, t)),
    ("deep_learning", "plot_regression_fit", lambda hp: hp.plot_regression_fit(x, t, x_dense, t_dense, y_true=t_dense, residuals=True, band=(t_dense - 0.2, t_dense + 0.2))),
    ("deep_learning", "plot_conditional_density", lambda hp: hp.plot_conditional_density(x_dense, t_dense, 0.5, torch.linspace(-1, 1, 50), torch.exp(-torch.linspace(-1, 1, 50) ** 2), 0.0)),
    ("deep_learning", "plot_polynomial_extrapolation", lambda hp: hp.plot_polynomial_extrapolation(x, t, x_dense, t_dense, h_grid=t_dense, degree=3)),
    ("deep_learning", "plot_dataset_size_remedy", lambda hp: hp.plot_dataset_size_remedy(x_dense, t_dense, [(20, x, t, t_dense), (20, x, t, t_dense)], M=9)),
    ("deep_learning", "plot_bias_variance_curve", lambda hp: hp.plot_bias_variance_curve([0.1, 1.0, 10.0], [0.1, 0.05, 0.01], [0.01, 0.05, 0.1], [0.11, 0.1, 0.11])),
    ("deep_learning", "plot_ensemble_fits", lambda hp: hp.plot_ensemble_fits(x_dense, t_dense, {k: ([t_dense] * 3, t_dense) for k in ("a", "b", "c")})),
    ("deep_learning", "plot_least_squares_geometry", lambda hp: hp.plot_least_squares_geometry(
        torch.tensor([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]), torch.tensor([1.2, 0.3, 1.8]),
        torch.tensor([0.8, 1.1, 1.4]), torch.tensor([0.8, 0.6]))),
    ("deep_learning", "plot_time_series", lambda hp: hp.plot_time_series(list(range(10)), list(range(10)), label="s")),
    ("deep_learning", "plot_attention_heatmap", lambda hp: hp.plot_attention_heatmap(np.full((2, 3), 1 / 3), ["a", "b", "c"],
        query_tokens=["x", "y"], show_values=False, title="t")),
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
