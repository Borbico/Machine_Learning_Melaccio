import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import classification_report

columns = [
    "class",
    "a1", "a2", "a3", "a4", "a5", "a6",
    "id"
]

feature_names = columns[1:-1]
labels = [0, 1]

set_base_path = 'data/monk/monks-{}.{}'


def read_file(monk_path: str) -> DataFrame:
    """
    Read a file into a DataFrame
    :param monk_path: the monk file path
    :return: a DataFrame containing the original data
    """

    orig = pd.read_csv(monk_path, sep=r"\s+", header=None, names=columns)
    return orig


def load_set(nr: int) -> (DataFrame, DataFrame):
    """
    Load the monk training and testing sets by passing the monk set id, such as 1,2 or 3
    :param nr: int the number of sets (1, 2, 3)
    :return: the corresponding (1,2, or 3) training and testing sets as DataFrames
    """
    monk_tr_path = set_base_path.format(nr, "train")
    monk_ts_path = set_base_path.format(nr, "test")

    return read_file(monk_tr_path), read_file(monk_ts_path)


def split_and_prepare_dataset(df_orig: DataFrame) -> (DataFrame, ndarray):
    """
    Split dataset into training and testing sets eventually applying one-hot encoding
    :param df_orig: the original dataset
    :return: A tuple (X, y) representing the features and the labels
    """

    # Extract 'class' column which will be our labels
    y = df_orig["class"].to_numpy()

    # Drop class and id columns to extract our set of features
    X = df_orig.drop(columns=["class", "id"]) # The features
    # Transform features into a One-hot encoded training set
    # from [a1, a2, ..., an] to [a1_1, a1_2, a1_3, a2_1, a1_2, a2_3,..., an_1, an_2, an_3]
    X_onehot_encoded = pd.get_dummies(X, columns=pd.Index(feature_names), dtype=int)
    return X_onehot_encoded, y


def _class_counts_str(df, target="class") -> str:
    """
    Support class to be used by dataset_introspection
    :param df:
    :param target:
    :return:
    """
    vc = df[target].value_counts().sort_index()
    return ", ".join([f"class {k} = {v}" for k, v in vc.items()])


def dataset_introspection(df_TR: DataFrame, df_TS: DataFrame) -> DataFrame:
    """
    Print a summary of the dataset introspection
    :param df_TR: the training dataframe
    :param df_TS: the testing dataframe
    :return: a summary of the dataset introspection in form of a DataFrame
    """

    dataset_overview = pd.DataFrame({
        "Property": [
            "Number of samples",
            "Number of features",
            "Class values",
            "Class balance"
        ],
        "Training": [
            df_TR.shape[0],
            df_TR.shape[1] - 2,
            sorted(df_TR["class"].unique().tolist()),
            _class_counts_str(df_TR)
        ],
        "Test": [
            df_TS.shape[0],
            df_TS.shape[1] - 2,
            sorted(df_TS["class"].unique().tolist()),
            _class_counts_str(df_TS)
        ]
    })

    return dataset_overview


def analyze_lr_curve(history, early_epoch=30):
    """
    Best curve
    :param history: Can be vl_mee history or vl_mse history etc..
    :param early_epoch: the epoch in which the curve descend more rapidly
    :return:
    """
    arr = np.asarray(history, dtype=float)

    if len(arr) == 0:
        return {
            "early_gain": np.nan,
            "oscillation": np.nan,
            "best_vl_metric": np.nan,
            "best_epoch": np.nan
        }

    if len(arr) == 1:
        return {
            "early_gain": 0.0,
            "oscillation": 0.0,
            "best_vl_metric": float(arr[0]),
            "best_epoch": 1
        }

    k = min(early_epoch - 1, len(arr) - 1)

    start = arr[0]
    early = arr[k]
    best = np.min(arr)
    best_epoch = int(np.argmin(arr) + 1)

    # best descent in a specific epoch frame
    # If metric is vl mee we are basically doing:
    # early gain = (vl_mee(1) - vl_mee(k)) / vl_mee(1)
    early_gain = (start - early) / max(abs(start), 1e-12)

    # instability of the curve calculated or its oscillation
    # oscillation = std( Delta of vl_mee(t) )
    # where Delta of vl_mee(t) equals to:
    # Delta vl_mee(t) = vl_mee(t) - vl_mee(t-1)
    diffs = np.diff(arr)
    oscillation = np.std(diffs)

    return {
        "early_gain": float(early_gain),
        "oscillation": float(oscillation),
        "best_vl_metric": float(best),
        "best_epoch": best_epoch
    }


def evaluate_lr(fold_results, epoch_metric:int, fold_metric:str, early_epoch:int) -> tuple[float, float,float]:
    """
    Extract learning rate and other metrics from fold results for being used in optuna study evaluation.
    The returned tuple contains the following in the same exact order:
        - mean_best: the mean main metric
        - mean_oscillation
        - mean_early_gain
    :param fold_results: the fold history as returned by kfold
    :param epoch_metric: for example epochs_vl_mee
    :param fold_metric: for example fold_vl_mee
    :param early_epoch:
    :return: summary["mean_best"],summary["mean_oscillation"],summary["mean_early_gain"]
    """

    analyses = [analyze_lr_curve(getattr(fold, epoch_metric), early_epoch=early_epoch) for fold in fold_results]
    summary = {
        "mean_early_gain": float(np.mean([a["early_gain"] for a in analyses])),
        "std_early_gain": float(np.std([a["early_gain"] for a in analyses])),

        "mean_oscillation": float(np.mean([a["oscillation"] for a in analyses])),
        "std_oscillation": float(np.std([a["oscillation"] for a in analyses])),

        "mean_best": float(np.mean([getattr(fold, fold_metric) for fold in fold_results])),
        "std_best": float(np.std([getattr(fold, fold_metric) for fold in fold_results])),

        "mean_best_epoch": float(np.mean([a["best_epoch"] for a in analyses]))
    }
    # the order that will be evaluated according to find_best_lr_from_trials
    return summary["mean_best"],summary["mean_oscillation"]


def find_best_lr_from_trials(study) -> float:
    """
    The learning rate is selected according to three complementary criteria derived from the training dynamics observed during K-Fold cross-validation:
	    - Stable optimization: the learning curve should exhibit low oscillation, suggesting that the step size is not too large and the optimization process is stable.
	    - Best validation performance: among the candidate values, preference is given to the learning rate achieving the lowest mean validation metric across the K-Fold splits.
    :param study: the study object
    :return: the best learning rate
    """

    results = min(
        study.best_trials,
        key=lambda t: (
            t.values[0],    # 1. minimize main metric as first step
            t.values[1],    # 2. oscillation as second step in case two or more trials share the same minimum (1)
        )
    )

    return results.params["learning_rate"]


def classification_helper(y_pred):
    return y_pred[2].detach().cpu().numpy().ravel().astype(int)
