# ============================================================
# KNN Functions
# ============================================================

from sklearn.model_selection import cross_val_score
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
        # Note: at each fold iteration scikit-learn create a brand-new model
        # based on model_k we are passing to the function
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
    plt.plot(k_values, train_scores, marker="o", label="Training accuracy")
    plt.plot(k_values, cv_scores, marker="o", label="CV mean accuracy")
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