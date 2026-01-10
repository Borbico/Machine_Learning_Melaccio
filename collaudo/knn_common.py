# ============================================================
# KNN Functions
# ============================================================
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

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


def knn_introspection(cv_scores: list) -> (int, float, float):
    """
    Print and return the best k value and its mean and standard deviation.
    This function is basically a wrapper of extract_best_knn_metrics
    :param cv_scores: the cross validation scores
    :return: best_k, best_mean, best_std
    """

    best_k, best_mean, best_std = extract_best_knn_metrics(cv_scores)
    print("Best k:", best_k)
    print("Best mean accuracy:", best_mean)
    print("Std:", best_std)

    return best_k, best_mean, best_std


def plot_knn_learning_curve(model, X, y, cv, ax=None):
    """
    Plot a learning curve for a given KNN model.
    Can be reused inside subplot grids if ax is provided.
    """
    k = model.n_neighbors
    random_state = cv.random_state

    y_np = np.asarray(y).ravel()

    # compute safe train sizes
    frac_train_fold = 1 - 1 / cv.get_n_splits()
    min_frac = (k + 1) / (frac_train_fold * len(y_np))
    min_frac = max(min_frac, 0.1)
    min_frac = min(min_frac, 0.9)

    train_sizes = np.linspace(min_frac, 1.0, 15)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X, y_np,
        cv=cv,
        scoring="accuracy",
        train_sizes=train_sizes,
        shuffle=True,
        random_state=random_state
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # use provided axis or create one
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(train_sizes_abs, train_mean, label="Training accuracy")
    ax.plot(train_sizes_abs, val_mean, label="CV accuracy")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"K-NN learning curve (k={k})")
    ax.legend()
    ax.grid(True)


def plot_knn_learning_curves_grid(models, X, y, cv, n_cols=2, highlight_k=None):
    """
    Plot learning curves for multiple KNN models in a grid layout.
    Optionally highlight the subplot whose model.n_neighbors == highlight_k.
    """
    import numpy as np
    import matplotlib.pyplot as plt

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
        plot_knn_learning_curve(model=model, X=X, y=y, cv=cv, ax=ax)

        # highlight if requested
        if highlight_k is not None and model.n_neighbors == highlight_k:
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
    - always includes best_k
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


