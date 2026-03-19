# ============================================================
# KNN Functions
# ============================================================


import numpy as np
from networkx.classes import non_neighbors
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt
import cross_common as cr



def extract_best_knn_metrics_from_grid(gs):

    best_model = gs.best_estimator_.named_steps["knn"]
    clean_params = {
        param.split("__")[-1]: value
        for param, value in gs.best_params_.items()
    }

    return best_model, clean_params

N_NEIGHBORS = "n_neighbors"


def label_row(r, step_name:str="model"):
    k = r.get(f"param_{step_name}__n_neighbors", "")
    w = r.get(f"param_{step_name}__weights", "")
    m = r.get(f"param_{step_name}__metric", "")
    p = r.get(f"param_{step_name}__p", "")
    return f"k={k} | w={w}, m={m}, p={p}"


def plot_knn_validation_curve_from_gs(gs, agg:str="max", score_correction:int=1, step_name:str="model"):

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
        elif agg == "mean":
            agg_test.append(np.mean(test_scores[mask]))
            agg_train.append(np.mean(train_scores[mask]) if not np.isnan(train_scores[mask]).all() else np.nan)
        else:
            raise ValueError("agg must be 'max' or 'mean'")

    uniq_k = np.array(uniq_k, dtype=int)
    agg_test = np.array(agg_test, dtype=float)
    agg_train = np.array(agg_train, dtype=float)

    # 3) accuracy vs neg_*
    scoring = gs.scoring
    y_label = "Score"
    plot_train = not np.isnan(agg_train).all()

    # Change Y axes according to scoring
    if isinstance(scoring, str):
        if scoring == "accuracy":
            y_label = "Accuracy"
        elif scoring.startswith("neg_"):

            if "mean_squared_error" in scoring:
                y_label = "MSE"
            elif "mean_absolute_error" in scoring:
                y_label = "MAE"
            elif "root_mean_squared_error" in scoring:
                y_label = "RMSE"
            else:
                y_label = scoring.replace("neg_", "").replace("_", " ").upper()

    # 4) Plot
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


def plot_knn_validation_curve(cv_scores, train_scores):

    k_values = [k[0] for k in cv_scores]
    cv_scores = [c[1] for c in cv_scores]
    train_scores = [t[1] for t in train_scores]
    plt.figure()
    plt.plot(k_values, train_scores, marker="o", ms=1, label="Training accuracy")
    plt.plot(k_values, cv_scores, marker="o", ms=1, label="CV mean accuracy")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Accuracy")
    plt.title("K-NN: training vs CV (overfitting check)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_knn_validation_curve_regression(k_values, train_mse, cv_mse):
    plt.figure()
    plt.plot(k_values, train_mse, marker="o", ms=3, label="Training MSE")
    plt.plot(k_values, cv_mse, marker="o", ms=3, label="CV mean MSE")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("MSE")
    plt.title("K-NN Regressor: training vs CV (overfitting check)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_knn_learning_curve(model, X, y, cv, scoring, ax=None):
    """
    Plot a learning curve for a given KNN model (classifier or regressor).
    If scoring is a neg_* loss, the curve is plotted as the corresponding positive loss.
    """
    k = model.n_neighbors
    random_state = getattr(cv, "random_state", None)

    # compute safe train sizes (avoid k > n_train_fold)
    frac_train_fold = 1 - 1 / cv.get_n_splits()
    min_frac = (k + 1) / (frac_train_fold * len(y))
    min_frac = min(max(min_frac, 0.1), 0.9)
    train_sizes = np.linspace(min_frac, 1.0, 15)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        shuffle=True,
        random_state=random_state
    )

    train_mean, val_mean = cr.apply_score_correction(scoring,(train_scores.mean(axis=1),val_scores.mean(axis=1)))

    # Decide label + possible sign flip
    y_label = "Score"
    train_label = "Training score"
    val_label = "CV score"
    title_metric = scoring

    if isinstance(scoring, str) and scoring.startswith("neg_"):

        if scoring == "neg_mean_absolute_error":
            y_label = "MAE"
            title_metric = "MAE"
        elif scoring == "neg_mean_squared_error":
            y_label = "MSE"
            title_metric = "MSE"
        elif scoring == "neg_root_mean_squared_error":
            y_label = "RMSE"
            title_metric = "RMSE"
        else:
            y_label = scoring.replace("neg_", "").replace("_", " ").upper()
            title_metric = y_label

        train_label = f"Training {y_label}"
        val_label = f"CV {y_label}"
    elif scoring == "accuracy":
        y_label = "Accuracy"
        train_label = "Training accuracy"
        val_label = "CV accuracy"
        title_metric = "Accuracy"

    # plot
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(train_sizes_abs, train_mean, label=train_label)
    ax.plot(train_sizes_abs, val_mean, label=val_label)

    ax.set_xlabel("Training set size")
    ax.set_ylabel(y_label)
    ax.set_title(f"K-NN learning curve (k={k})")
    ax.legend()
    ax.grid(True)


def plot_knn_learning_curves_grid(model, X, y, cv, scoring, n_cols=2):
    """
    Plot learning curves for multiple KNN models in a grid layout.
    Optionally highlight the subplot whose model.n_neighbors == highlight_k.
    """
    models = []
    model_params = model.get_params()
    Estimator = type(model)

    kn=model_params["n_neighbors"]
    for k in k_neighborhood(kn, len(y)):
        params = {**model_params, "n_neighbors": k}
        models.append(Estimator(**params))

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

        # highlight if requested
        if model.n_neighbors == kn:
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)   # bordo più spesso
            ax.set_title(f"K-NN learning curve (k={model.n_neighbors})  *")

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
