# ============================================================
# KNN Functions
# ============================================================
import warnings

import numpy as np
from sklearn import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


def extract_best_knn_metrics_from_grid(gs):

    best_model = gs.best_estimator_.named_steps["knn"]
    clean_params = {
        param.split("__")[-1]: value
        for param, value in gs.best_params_.items()
    }

    return best_model, clean_params


def extract_best_knn_metrics(cv_scores: list) -> tuple:
    """
    Find the best scores
    :param cv_scores: the cross validation scores
    :return: a tuple containing the best K, score, and standard deviation
    """

    return max(cv_scores, key=lambda x: x[1])


def find_best_knn_model(X, y, cv, k_range: list, scoring:str = "accuracy", weights:str = "uniform", metric:str = "euclidean") -> (KNeighborsClassifier, list):
    """
    Find the best KNN model using cross validation
    :param X:
    :param y:
    :param cv:
    :param k_range:
    :param scoring:
    :param weights:
    :param metric:
    :return:
    """

    warnings.warn(
        "old_function is deprecated and will be removed in a future release. "
        "Use new_function instead.",
        category=DeprecationWarning,
        stacklevel=2
    )

    train_scores = [] # Train scores
    cv_scores = [] # Cross validation scores

    # iterate over K
    for k in k_range:

        # 1.Instantiate our KNN classifier
        model_k = KNeighborsClassifier(
            n_neighbors=k,
            weights=weights,
            metric=metric
        )

        # 2. Calculate the mean accuracy on the given test data and labels.
        model_k.fit(X, y)
        score = model_k.score(X, y)
        train_scores.append((k, score))

        # 3. Evaluate the scores by cross-validation.
        # Note: as stated in scikit-learn documentation
        # at each fold iteration the function creates a brand-new model
        # based on model_k passed in input to the function
        scores = cross_val_score(model_k, X, y, cv=cv, scoring=scoring)
        mean = scores.mean() # Mean accuracy
        std = scores.std() # Standard deviation
        cv_scores.append((k,mean,std))

    # Retrieve the best classifier
    best_k, best_mean, best_std = extract_best_knn_metrics(cv_scores)

    # Finally build a classifier with best K
    best_model = KNeighborsClassifier(
        n_neighbors=best_k,
        weights=weights,
        metric=metric
    )

    return best_model, cv_scores, train_scores


def plot_knn_validation_curve_from_gs_old(gs):
    """
    Plot training vs CV accuracy from a fitted GridSearchCV object
    """

    results = gs.cv_results_

    k_values = results["param_knn__n_neighbors"]
    train_scores = results["mean_train_score"]
    cv_scores = results["mean_test_score"]


    plt.figure(figsize=(7, 4))
    plt.plot(k_values, train_scores, label="Training accuracy")
    plt.plot(k_values, cv_scores, label="CV mean accuracy")

    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Accuracy")
    plt.title("K-NN validation curve (training vs CV)")
    plt.grid(True)
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_knn_validation_curve_from_gs(gs, agg:str="max"):

    r = gs.cv_results_
    best_k = gs.best_params_["knn__n_neighbors"]

    # 1) Estrai k (qui tieni la tua chiave esplicita; se vuoi la rendo automatica)
    k_vals = np.array(r["param_knn__n_neighbors"], dtype=int)

    test_scores = np.array(r["mean_test_score"], dtype=float)
    train_scores = np.array(r.get("mean_train_score", np.full_like(test_scores, np.nan)), dtype=float)

    # 2) Aggrega per k
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
            # Loss switched to positive for better readability
            agg_test = -agg_test
            agg_train = -agg_train

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


def plot_knn_learning_curve(model, X, y, cv, ax=None, scoring: str = "accuracy"):
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

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # Decide label + possible sign flip
    y_label = "Score"
    train_label = "Training score"
    val_label = "CV score"
    title_metric = scoring

    if isinstance(scoring, str) and scoring.startswith("neg_"):
        # Convert from negative score to positive loss for readability
        train_mean = -train_mean
        val_mean = -val_mean

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
    ax.set_title(f"K-NN learning curve (k={k}) – {title_metric}")
    ax.legend()
    ax.grid(True)


def plot_knn_learning_curves_grid(model, X, y, cv, n_cols=2, scoring="accuracy"):
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
            ax.set_title(f"K-NN learning curve (k={model.n_neighbors})  ★")

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


def regression_report(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Regression report — {name}")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2
    }


def mee(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))


import numpy as np
import pandas as pd
from sklearn.base import clone

def mee(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

def mee(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def run_kfold(untrained_base_model,X,y,fold_strategy):
    """
    K-Fold CV for sklearn models (multi-output regression) collecting:
    - MEE
    - MSE (averaged over outputs)
    - RMSE

    'best' and 'last' are defined over folds (no epochs in sklearn KNN):
    - best = fold with minimum MEE
    - last = last fold executed
    """

    # Convert to numpy once
    Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    yn = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    fold_rows = []

    for fold_nr, (tr_idx, vl_idx) in enumerate(fold_strategy.split(Xn), start=1):
        model = clone(untrained_base_model)

        X_tr, y_tr = Xn[tr_idx], yn[tr_idx]
        X_vl, y_vl = Xn[vl_idx], yn[vl_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_vl)

        fold_mee = mee(y_vl, pred)
        fold_mse = float(mean_squared_error(y_vl, pred))   # avg over 4 outputs
        fold_rmse = float(np.sqrt(fold_mse))

        fold_rows.append({
            "fold": fold_nr,
            "mee": fold_mee,
            "mse": fold_mse,
            "rmse": fold_rmse
        })

    per_fold = pd.DataFrame(fold_rows)

    # Choose "best" fold based on MEE (you can switch criterion if you prefer)
    best_idx = int(per_fold["mee"].idxmin())

    results = {
        "per_fold": per_fold,

        "mee_mean": float(per_fold["mee"].mean()),
        "mee_std":  float(per_fold["mee"].std(ddof=0)),
        "mse_mean": float(per_fold["mse"].mean()),
        "mse_std":  float(per_fold["mse"].std(ddof=0)),
        "rmse_mean": float(per_fold["rmse"].mean()),
        "rmse_std":  float(per_fold["rmse"].std(ddof=0)),

        "best_fold": int(per_fold.loc[best_idx, "fold"]),
        "mee_best":  float(per_fold.loc[best_idx, "mee"]),
        "mse_best":  float(per_fold.loc[best_idx, "mse"]),
        "rmse_best": float(per_fold.loc[best_idx, "rmse"]),

        "last_fold": int(per_fold.iloc[-1]["fold"]),
        "mee_last":  float(per_fold.iloc[-1]["mee"]),
        "mse_last":  float(per_fold.iloc[-1]["mse"]),
        "rmse_last": float(per_fold.iloc[-1]["rmse"]),
    }

    return results