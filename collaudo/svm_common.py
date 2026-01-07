import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve


def compare_svfreq_vs_permutation(best_model, X_tr, y_tr, scoring="accuracy"):
    # ---- 1) SV-frequency ----
    best_model.fit(X_tr, y_tr)
    svm = best_model.named_steps["svm"]
    SV = X_tr.iloc[svm.support_]
    sv_freq = SV.mean().sort_values(ascending=False)  # per colonna one-hot

    # ---- 2) Permutation importance ----
    perm = permutation_importance(
        best_model,
        X_tr, y_tr,
        scoring=scoring,
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )
    perm_imp = pd.Series(perm.importances_mean, index=X_tr.columns).sort_values(ascending=False)

    # ---- 3) Aggregazione per attributo (a1..a6) ----
    groups = X_tr.columns.to_series().astype(str).str.split("_", n=1).str[0]

    # Per SV-frequency: usa MAX (evita l'artefatto del sum=1.0)
    sv_attr = sv_freq.groupby(groups).max().sort_values(ascending=False)

    # Per permutation: somma dei contributi delle dummy dell'attributo
    perm_attr = perm_imp.groupby(groups).sum().sort_values(ascending=False)

    # ---- 4) Tabella confronto per attributo ----
    comp_attr = pd.concat(
        [sv_attr.rename("SV_freq_max"), perm_attr.rename("Perm_importance_sum")],
        axis=1
    ).sort_values("Perm_importance_sum", ascending=False)

    return sv_freq, perm_imp, comp_attr


def plot_learning_curve(estimator, X, y, cv):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    # Calcolo media e deviazione standard
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

    # Area della deviazione standard (opzionale, per vedere la varianza)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    plt.title("Learning curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def plot_learning_curve_svm(model, X, y, cv, scoring:str="accuracy"):

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 15),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_mean, label="Train")
    plt.plot(train_sizes, val_mean, label="CV")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_curve_svr(model, X, y, cv, scoring: str = "neg_mean_squared_error"):
    """
    Learning curve for SVR (regression).

    If a neg_* scorer is used (e.g. neg_mean_squared_error),
    the sign is flipped for visualization so that
    lower values correspond to better performance.
    """

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 15),
        n_jobs=-1
    )

    # Mean over folds
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # Handle neg_* scorers (typical for regression)
    if scoring.startswith("neg_"):
        train_mean = -train_mean
        val_mean = -val_mean
        ylabel = scoring.replace("neg_", "") + " (lower is better)"
    else:
        ylabel = scoring + " (higher is better)"

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, label="Train")
    plt.plot(train_sizes, val_mean, label="CV")
    plt.xlabel("Training set size")
    plt.ylabel(ylabel)
    plt.title("Learning Curve (SVR)")
    plt.legend()
    plt.grid(True)
    plt.show()


def grid_introspection(grid):

    print("Total model explored:", len(grid.cv_results_["params"]))
    print("Best params:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)
    print("Best model:", grid.best_estimator_)

    svm = grid.best_estimator_.named_steps["svm"]
    n_sv = svm.n_support_.sum()
    print("Total support vectors:", n_sv)
    print("Support vectors per class:", svm.n_support_)


def label_row(r):
    k = r.get("param_svm__kernel", "")
    C = r.get("param_svm__C", "")
    g = r.get("param_svm__gamma", "")
    d = r.get("param_svm__degree", "")
    # Mostra solo i parametri rilevanti per quel kernel
    if k == "linear":
        return f"{k} | C={C}"
    elif k == "rbf":
        return f"{k} | C={C}, γ={g}"
    elif k == "poly":
        return f"{k} | C={C}, γ={g}, deg={d}"
    elif k == "sigmoid":
        return f"{k} | C={C}, γ={g}"
    else:
        return f"{k} | C={C}, γ={g}, deg={d}"


def plot_results_svc(results, TOP_N=10, scoring_name="balanced_accuracy"):
    res = pd.DataFrame(results)

    # Ordina: migliore → peggiore
    res_sorted = res.sort_values(["mean_test_score", "rank_test_score"], ascending=[False, True]).reset_index(drop=True)

    # Top e Worst
    top = res_sorted.head(TOP_N).copy()
    worst = res_sorted.tail(TOP_N).copy().iloc[::-1].copy()  # inverti: peggiore in alto

    # Etichette (richiede che label_row esista già)
    top["model_label"] = top.apply(label_row, axis=1)
    worst["model_label"] = worst.apply(label_row, axis=1)

    # --- Plot TOP ---
    plt.figure(figsize=(9, 3))
    scores = top["mean_test_score"].astype(float).values
    errs = top["std_test_score"].astype(float).values
    plt.barh(top["model_label"], scores, xerr=errs)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Mean CV score ({scoring_name})")
    plt.title(f"Top {TOP_N} models")
    plt.tight_layout()
    plt.show()

    # --- Plot WORST ---
    plt.figure(figsize=(9, 3))
    scores = worst["mean_test_score"].astype(float).values
    errs = worst["std_test_score"].astype(float).values
    plt.barh(worst["model_label"], scores, xerr=errs)
    plt.gca().invert_yaxis()  # peggiore in alto
    plt.xlabel(f"Mean CV score ({scoring_name})")
    plt.title(f"Worst {TOP_N} models from GridSearchCV")
    plt.tight_layout()
    plt.show()


def plot_results_svr(results, TOP_N: int = 10, scoring_name: str | None = None):
    """Plot the top-N SVR models from a GridSearchCV/RandomizedSearchCV.

    This helper is tailored for regression scorers. In scikit-learn, losses are
    typically provided as `neg_*` scorers to preserve the convention
    "higher is better". Here we flip the sign for display and sort by the
    *lowest* (positive) loss.

    Parameters
    ----------
    results : dict | object
        Either the `cv_results_` dict or a fitted search object exposing
        `.cv_results_` and optionally `.scoring`.
    TOP_N : int
        Number of top configurations to show.
    scoring_name : str | None
        Name of the scorer (e.g., 'neg_mean_squared_error'). If None and a
        fitted search object is passed, it is inferred from `search.scoring`.

    Notes
    -----
    - If `scoring_name` starts with 'neg_', the plot will show the corresponding
      positive loss and annotate "lower is better".
    - If it does not start with 'neg_', the plot behaves like a standard score
      (higher is better).
    """
    # Accept either cv_results_ dict or a fitted search object
    if hasattr(results, "cv_results_"):
        if scoring_name is None:
            scoring_name = getattr(results, "scoring", None)
        cv = results.cv_results_
    else:
        cv = results

    if not isinstance(cv, dict) or "mean_test_score" not in cv:
        raise ValueError("plot_results_svr expects a cv_results_ dict or a fitted search object with cv_results_.")

    res = pd.DataFrame(cv)

    # Infer neg-loss if possible
    is_neg_loss = isinstance(scoring_name, str) and scoring_name.startswith("neg_")
    if scoring_name is None:
        # heuristic: if all scores are <= 0, likely a neg-loss
        s = res["mean_test_score"].astype(float)
        is_neg_loss = (s.max() <= 0) and (s.min() < 0)

    # Convert to display score and choose sorting direction
    if is_neg_loss:
        res["display_score"] = -res["mean_test_score"].astype(float)
        sort_ascending = True  # lower loss is better
        metric_label = (scoring_name or "loss").replace("neg_", "")
        better_note = "lower is better"
    else:
        res["display_score"] = res["mean_test_score"].astype(float)
        sort_ascending = False  # higher score is better
        metric_label = scoring_name or "score"
        better_note = "higher is better"

    # Sort best -> worst
    sort_cols = ["display_score"]
    if "rank_test_score" in res.columns:
        # rank_test_score is always based on mean_test_score, but it can help tie-breaking
        sort_cols.append("rank_test_score")

    res_sorted = res.sort_values(sort_cols, ascending=[sort_ascending, True]).reset_index(drop=True)
    top = res_sorted.head(int(TOP_N)).copy()

    # Labels (requires label_row)
    if any(c.startswith("param_") for c in top.columns):
        top["model_label"] = top.apply(label_row, axis=1)
    else:
        # fallback if params are stored differently
        top["model_label"] = [f"model {i}" for i in range(len(top))]

    # Plot TOP
    plt.figure(figsize=(10, 3))
    scores = top["display_score"].astype(float).values
    errs = top.get("std_test_score", pd.Series([0.0] * len(top))).astype(float).values
    plt.barh(top["model_label"], scores, xerr=errs)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Mean CV {metric_label} ({better_note})")
    plt.title(f"Top {len(top)} SVR models")
    plt.tight_layout()
    plt.show()


def regression_report(y_true, y_pred):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = np.abs(df["error"])
    df["sq_error"] = df["error"] ** 2

    report = (
        df
        .groupby("y_true")
        .agg(
            support=("y_true", "count"),
            mean_pred=("y_pred", "mean"),
            MAE=("abs_error", "mean"),
            RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
            bias=("error", "mean")
        )
        .reset_index()
        .rename(columns={"y_true": "true_quality"})
    )

    print(report)

    return

def plot_true_vs_preds_svr(y_true, y_pred):

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        '--', linewidth=2
    )

    plt.xlabel("True quality")
    plt.ylabel("Predicted quality")
    plt.title("SVR – Cross-validated predictions")
    plt.grid(True)
    plt.show()
