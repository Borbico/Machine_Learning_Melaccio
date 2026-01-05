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


# def plot_learning_curve(grid_monk1.best_estimator_, X_train_monk1, y_train_monk1, "Learning Curve (MONK-1)")
def plot_learning_curve_svm(model, X, y, cv):

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=cv,
        scoring="balanced_accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_mean, marker="o", label="Train")
    plt.plot(train_sizes, val_mean, marker="o", label="CV")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
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


def plot_results(results, TOP_N=10, scoring_name="balanced_accuracy"):
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
    plt.title(f"Top {TOP_N} models from GridSearchCV")
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
