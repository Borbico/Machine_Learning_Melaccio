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
import cup_common as cc


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


def plot_knn_validation_curve_from_gs(gs, agg:str="max", score_correction:int=1):

    r = gs.cv_results_
    best_k = gs.best_params_["regressor__n_neighbors"] # best K
    k_vals = np.array(r["param_regressor__n_neighbors"], dtype=int) # all tried K

    test_scores = np.array(score_correction * r["mean_test_score"], dtype=float)
    train_scores = np.array(score_correction * r.get("mean_train_score", np.full_like(test_scores, np.nan)), dtype=float)

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


def plot_knn_learning_curve(model, X, y, cv, ax=None, score_correction:int=1, scoring: str = "accuracy"):
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

    train_mean = score_correction *  train_scores.mean(axis=1)
    val_mean = score_correction * val_scores.mean(axis=1)

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
    ax.set_title(f"K-NN learning curve (k={k})")
    ax.legend()
    ax.grid(True)


def plot_knn_learning_curves_grid(model, X, y, cv, n_cols=2, score_correction:int = 1, scoring="accuracy"):
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


# def mee(y_true, y_pred):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def run_kfold(untrained_base_model, X, y, fold_strategy, fill_value=np.nan) -> cc.FoldResults:
    """
    K-Fold CV for sklearn multi-output regression model (e.g., Pipeline+KNN).
    Returns fold_histories in a format aligned with your NN logs.

    Notes:
    - No epochs => per-epoch histories do not exist.
      We store single-point metrics per fold; mean/best/last coincide.
    - 'acc' fields are not defined for regression => filled with fill_value.
    - 'loss' fields: we map them to MSE (quadratic loss) for consistency.
    """

    Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    yn = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    fold_results = cc.FoldResults()

    for fold_nr, (tr_idx, vl_idx) in enumerate(fold_strategy.split(Xn), start=1):

        X_tr, y_tr = Xn[tr_idx], yn[tr_idx] # Train set
        X_vl, y_vl = Xn[vl_idx], yn[vl_idx] # Validation set

        # Clone model and fit
        model = clone(untrained_base_model)
        model.fit(X_tr, y_tr)

        # Predict on train-fold and val-fold
        pred_tr = model.predict(X_tr)
        pred_vl = model.predict(X_vl)

        # Mean squared error and root mean squared error
        tr_mse = float(mean_squared_error(y_tr, pred_tr))
        tr_rmse = float(np.sqrt(tr_mse))
        vl_mse = float(mean_squared_error(y_vl, pred_vl))
        vl_rmse = float(np.sqrt(vl_mse))
        # Mean euclidean error
        tr_mee, vl_mee = cc.mee(y_tr, pred_tr), cc.mee(y_vl, pred_vl)

        # Compared with NN we have no epochs, "history" is a single point.
        # std across epochs is 0.0, and best == last == that point.
        fold_results.append(cc.FoldResult({
            "fold_nr": fold_nr,

            # Map "loss" to MSE for regression (quadratic loss)
            "tr_loss": tr_mse, "tr_loss_std": 0.0,
            "vl_loss": vl_mse, "vl_loss_std": 0.0,
            "tr_rmse": tr_rmse, "vl_rmse": vl_rmse,

            # accuracy not defined in regression
            "tr_acc": fill_value, "tr_acc_std": fill_value,
            "vl_acc": fill_value, "vl_acc_std": fill_value,

            "tr_mae": fill_value, "tr_mae_std": fill_value,
            "vl_mae": fill_value, "vl_mae_std": fill_value,

            "tr_mee": tr_mee, "tr_mee_std": 0.0,
            "vl_mee": vl_mee, "vl_mee_std": 0.0,

            "best_tr_loss": tr_mse, "best_vl_loss": vl_mse,
            "best_tr_acc": fill_value, "best_vl_acc": fill_value,
            "best_tr_mee": tr_mee, "best_vl_mee": vl_mee
        }))

        # fold_results.append(cc.FoldResult({
        #     "fold_nr": fold_nr,
        #     "tr_loss": np.mean(hist_tr_loss), "tr_loss_std": np.std(hist_tr_loss),
        #     "vl_loss": np.mean(hist_vl_loss), "vl_loss_std": np.std(hist_vl_loss),
        #     "tr_acc": np.mean(tr_hist_acc), "tr_acc_std": np.std(tr_hist_acc),
        #     "vl_acc": np.mean(vl_hist_acc), "vl_acc_std": np.std(vl_hist_acc),
        #     "tr_mae": np.mean(hist_tr_mae), "tr_mae_std": np.std(hist_tr_mae),
        #     "vl_mae": np.mean(hist_vl_mae), "vl_mae_std": np.std(hist_vl_mae),
        #     "tr_mee": np.mean(hist_tr_mee), "tr_mee_std": np.std(hist_tr_mee),
        #     "vl_mee": np.mean(hist_vl_mee), "vl_mee_std": np.std(hist_vl_mee),
        #
        #     "best_tr_loss": float(min(hist_tr_loss)),
        #     "best_tr_acc": float(max(tr_hist_acc)),
        #
        #     "best_vl_loss": float(min(hist_vl_loss)),
        #     "best_vl_acc": float(max(vl_hist_acc)),
        #
        #     "best_tr_mee": float(min(hist_tr_mee)),
        #     "best_vl_mee": float(min(hist_vl_mee))
        # }))

    #return pd.DataFrame(fold_histories)
    return fold_results

