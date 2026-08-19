# ============================================================
# KNN Functions
# ============================================================


import numpy as np
from sklearn import clone
from matplotlib import pyplot as plt
import cross_common as cr


N_NEIGHBORS = "n_neighbors"


def gaussian_weights(distances) -> np.ndarray:
    """
    Custom function for experimenting the weights behaviour within grid search.
    :param distances: The distance matrix has shape (n_samples, k), where each row contains the distances between a query point and its k nearest neighbors.
            - distances = [
                [d(x₁, x₁₁), d(x₁, x₁₂), ..., d(x₁, x₁k)],
                [d(x₂, x₂₁), d(x₂, x₂₂), ..., d(x₂, x₂k)],
                ...
            ]
    :return: the gaussian distance
    """
    distances = np.asarray(distances)
    gamma = 0.1
    # the np.exp apply the function to
    # each element of the array
    return np.exp(-gamma * distances**2)


def gaussian_weights_wrapper(gamma: float):
    # def gaussian_weights(distances) -> np.ndarray:
    #     distances = np.asarray(distances)
    #     return np.exp(-gamma * distances**2)

    gaussian_weights.__name__ = f"gaussian_weights_gamma_{gamma}"
    return gaussian_weights


class GaussianWeights:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def __call__(self, distances):
        distances = np.asarray(distances)
        return np.exp(-self.gamma * distances**2)

    def __repr__(self):
        return f"GaussianWeights(gamma={self.gamma})"


def label_row(r, step_name:str="model"):
    k = r.get(f"param_{step_name}__n_neighbors", "")
    w = r.get(f"param_{step_name}__weights", "")
    m = r.get(f"param_{step_name}__metric", "")
    p = r.get(f"param_{step_name}__p", "")
    return f"k={k} | w={w}, m={m}, p={p}"


def plot_knn_validation_curve_from_gs(gs, y_label: str="Score", agg:str="max", step_name:str="model"):

    r = gs.cv_results_
    best_k = gs.best_params_[f"{step_name}__{N_NEIGHBORS}"] # best K
    k_vals = np.array(r[f"param_{step_name}__{N_NEIGHBORS}"], dtype=int) # all tried K

    test_scores = np.array(cr.apply_score_correction(gs.scoring, r["mean_test_score"]))
    train_scores = np.array(cr.apply_score_correction(gs.scoring, r["mean_train_score"]))

    # Group by K
    uniq_k = np.unique(k_vals)
    agg_test = []
    agg_train = []

    for k in uniq_k:
        mask = (k_vals == k)

        if agg == "max":
            agg_test.append(np.max(test_scores[mask]))
            agg_train.append(np.max(train_scores[mask]) if not np.isnan(train_scores[mask]).all() else np.nan)
        elif agg == "min":
            agg_test.append(np.min(test_scores[mask]))
            agg_train.append(np.min(train_scores[mask]) if not np.isnan(train_scores[mask]).all() else np.nan)
        elif agg == "mean":
            agg_test.append(np.mean(test_scores[mask]))
            agg_train.append(np.mean(train_scores[mask]) if not np.isnan(train_scores[mask]).all() else np.nan)
        else:
            raise ValueError("agg must be 'max' or 'mean'")

    uniq_k = np.array(uniq_k, dtype=int)
    agg_test = np.array(agg_test, dtype=float)
    agg_train = np.array(agg_train, dtype=float)

    plot_train = not np.isnan(agg_train).all()
    plt.figure()

    if plot_train:
        plt.plot(uniq_k, agg_train, label=f"Training ({agg})")

    plt.plot(uniq_k, agg_test, label=f"CV mean ({agg})")
    plt.axvline(int(best_k), linestyle="--", alpha=0.7, label=f"Best k = {best_k}")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel(y_label)
    plt.title(f"K-NN validation curve ({y_label}, aggregated by k, {agg})")

    plt.grid(True)
    plt.legend()
    plt.show()


def plot_knn_learning_curve(model, X, y, cv, ax=None, scoring=None, title:str=None):
    """
    Wrapper to cr.plot_learning_curve to avoid size error in plotting KNN learning curves.
    :param model: the model to plot
    :param X_tr: training data
    :param y_tr: training labels
    :param cv: cross-validation generator
    :return: None
    """

    selected_k = model.named_steps.model.n_neighbors

    max_train_size = min(
        len(train_indices)
        for train_indices, _ in cv.split(X, y)
    )

    safe_train_sizes = np.unique(
        np.linspace(
            selected_k,
            max_train_size,
            20,
            dtype=int
        )
    )
    cr.plot_learning_curve(model, X, y, cv, ax, scoring, title, train_sizes=safe_train_sizes)


def plot_knn_learning_curves_grid(model, X, y, cv, scoring, n_cols=2, step_name:str="model"):
    """
    Plot learning curves for multiple KNN models in a grid layout.
    Optionally highlight the subplot whose model.n_neighbors == highlight_k.
    """
    models = []
    model_params = model.get_params()[step_name]

    kn=model_params.n_neighbors
    for k in k_neighborhood(kn, len(y)):
        clean_pipeline = clone(model)
        clean_pipeline.named_steps["model"].n_neighbors = k
        models.append(clean_pipeline)

    n_plots = len(models)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]

        # draw the learning curve on this axis
        plot_knn_learning_curve(model=model, X=X, y=y, cv=cv, ax=ax, scoring=scoring)
        #cr.plot_learning_curve(model=model, X=X, y=y, cv=cv, ax=ax, scoring=scoring)

        # highlight if requested
        if model.named_steps["model"].n_neighbors == kn:
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)   # bordo più spesso
            #ax.set_title(f"K-NN learning curve (k={model.named_steps["model"].n_neighbors})  *")
        ax.set_title(f"Learning curve (k={model.named_steps["model"].n_neighbors})")

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def k_neighborhood(best_k: int,n_samples: int,step: int = 2,min_k: int = 1) -> list:
    """
    Return exactly 6 reasonable k values around best_k
    for analysis purposes.

    The returned list:
    - always includes best_k as a central value when applicable
    - contains only odd k (step=2)
    - is clipped to [min_k, n_samples]
    """

    # candidate offsets to get 6 values (best_k in the middle if possible)
    offsets = [-5, -3, -1, 1, 3, 5]

    k_values = [
        best_k + offset * step
        for offset in offsets
    ]

    # include best_k explicitly
    k_values.append(best_k)

    # filter valid k
    k_values = [
        k for k in k_values
        if min_k <= k <= n_samples
    ]

    # remove duplicates and sort
    k_values = sorted(set(k_values))

    # if we have more than 6, keep the 6 closest to best_k
    if len(k_values) > 6:
        k_values = sorted(
            k_values,
            key=lambda k: abs(k - best_k)
        )[:6]
        k_values = sorted(k_values)

    # if we have fewer than 6 (edge cases near boundaries)
    while len(k_values) < 6:
        candidate = k_values[-1] + step
        if candidate <= n_samples:
            k_values.append(candidate)
        else:
            break

    return k_values
