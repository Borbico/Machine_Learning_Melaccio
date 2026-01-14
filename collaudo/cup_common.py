import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial.distance import euclidean
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


output_columns = ["t1", "t2", "t3", "t4"]
base_columns = [
    "id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"
]

tr_columns = base_columns + output_columns


#feature_names = columns[1:-1]
#labels = [0, 1]

set_base_path = 'data/cup/ML-CUP25-{}.csv'

def _read_file(cup_path: str) -> DataFrame:
    """
    Read a file into a DataFrame
    :param monk_path: the monk file path
    :return: a DataFrame containing the original data
    """
    orig = pd.read_csv(cup_path, sep=r",", header=None, skiprows=7)
    return orig


def load_set() -> (DataFrame, DataFrame):
    """
    Load the monk training and testing sets by passing the monk set id, such as 1,2 or 3
    :return: the corresponding red and wine as DataFrames
    """
    tr_path = set_base_path.format("TR")
    ts_path = set_base_path.format("TS")

    # Load dataframe and add a specific
    # type column to each set
    df_tr_orig = _read_file(tr_path)
    df_tr_orig.columns=tr_columns
    df_ts_orig = _read_file(ts_path)
    df_ts_orig.columns=base_columns

    return df_tr_orig, df_ts_orig


def prepare_dataset(df: DataFrame, scaler=None, fit_scaler: bool = False):
    """
    Prepare ML-CUP dataset:
    - Extract X (features) and y (targets, if present)
    - Optionally scale X using a provided scaler
      * fit_scaler=True  -> fit_transform (ONLY for training set)
      * fit_scaler=False -> transform (for validation/test set)
    Returns:
      X: pandas DataFrame (float32)
      y: numpy ndarray (float32) or None
    """

    # Extract X and y
    if np.isin(np.array(output_columns), df.columns.values).all():
        y = df[output_columns].to_numpy(dtype=np.float32)
        X_unscaled = df.drop(columns=["id"] + output_columns)
    else:
        y = None
        X_unscaled = df.drop(columns=["id"])

    # Ensure float32
    X_unscaled = X_unscaled.astype(np.float32)

    # Scale if requested
    if scaler is not None:
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_unscaled)
        else:
            X_scaled = scaler.transform(X_unscaled)

        X = pd.DataFrame(X_scaled, columns=X_unscaled.columns, index=X_unscaled.index)
    else:
        X = X_unscaled.copy()

    # Keep consistent indexing
    X = X.reset_index(drop=True)

    return X, y


def prepare_dataset_for_dry_run(df_train: DataFrame, ratio: float = 0.2, random_state:int = 42, scaler=None) -> (DataFrame, ndarray, DataFrame, ndarray, DataFrame, ndarray, DataFrame, ndarray):
    """
    Split dataset into training and testing sets eventually applying a scaler if present
    :param df_orig: the original dataset
    :param ratio: the ratio of training and testing sets (i.e 0.2 -> TR 80% VL 20%)
    :param random_state: the random state for shuffling
    :param scaler: the scaler to use for scaling
    :return: A tuple (X_tr, y_tr, X_ts, y_ts) representing the training and testing set
    """

    # divide dataframe in two df train and test based on ratio
    df_80, df_20 = train_test_split(
        df_train,
        test_size=ratio,
        random_state=random_state,
        shuffle=True
    )

    X_tr, y_tr = prepare_dataset(df_80, scaler=scaler, fit_scaler=True)
    X_ts, y_ts = prepare_dataset(df_20, scaler=scaler, fit_scaler=False)

    return X_tr, y_tr, X_ts, y_ts


def _class_counts_str(df, target="quality") -> str:
    """
    Support class to be used by dataset_introspection
    :param df:
    :param target:
    :return:
    """
    vc = df[target].value_counts().sort_index()
    return ", ".join([f"quality {k} = {v}" for k, v in vc.items()])


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
            [],#sorted(df_TR["quality"].unique().tolist()),
            [],#_class_counts_str(df_TR)
        ],
        "Test": [
            df_TS.shape[0],
            df_TS.shape[1] - 2,
            [],#sorted(df_TS["quality"].unique().tolist()),
            [] #_class_counts_str(df_TS)
        ]
    })

    return dataset_overview


def mee_summary(train_results: dict):

    vl_mee = np.asarray(train_results["hist_vl_mee"], dtype=float)

    best_epoch = int(np.argmin(vl_mee))          # epoca con MEE minimo
    best_vl_mee = float(vl_mee[best_epoch])      # valore minimo
    last_vl_mee = float(vl_mee[-1])              # ultimo valore (fine training)

    return {
        "best_epoch_mee": best_epoch,
        "best_vl_mee": best_vl_mee,
        "last_vl_mee": last_vl_mee
    }


def plot_kfold_bar_vl_mee(mee_history, use="best", ylabel="Validation MEE"):
    """
    Bar plot of validation MEE per fold + mean line.
    Uses keys: best_vl_mee / last_vl_mee.
    """
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    #key = f"{use}_vl_mee"
    #vals = np.array([h[key] for h in fold_histories], dtype=float)
    vals = np.array([h for h in mee_history], dtype=float)
    folds = np.arange(1, len(vals) + 1)
    mean_val = vals.mean()

    plt.figure(figsize=(7, 4))
    plt.bar(folds, vals)
    plt.axhline(mean_val, linestyle="--", linewidth=2, label=f"Mean = {mean_val:.4f}")
    plt.xticks(folds)
    plt.xlabel("Fold")
    plt.ylabel(ylabel)
    plt.title(f"K-Fold validation MEE per fold ({use})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kfold_bar_vl_rmse(mse, use="best"):
    """
    Bar plot of validation RMSE per fold + mean line.
    Assumes vl_loss stored is MSE (from MSELoss).
    """
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    key = f"{use}_vl_loss"
    rmse = np.sqrt(mse)

    folds = np.arange(1, len(rmse) + 1)
    mean_rmse = rmse.mean()

    plt.figure(figsize=(7, 4))
    plt.bar(folds, rmse)
    plt.axhline(mean_rmse, linestyle="--", linewidth=2, label=f"Mean = {mean_rmse:.4f}")
    plt.xticks(folds)
    plt.xlabel("Fold")
    plt.ylabel("Validation RMSE")
    plt.title(f"K-Fold validation RMSE per fold ({use})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def standard_metric(kfold_result):

    return (kfold_result["best_vl_mee"], kfold_result["vl_mee"], kfold_result["vl_mee_std"])


def mee_table(kfold_result, model_name="Model"):
    best_mee, mean_mee, mean_mee_std = standard_metric(kfold_result)

    df = pd.DataFrame(
        [[best_mee, mean_mee, mean_mee_std]],
        columns=["Best MEE", "Mean MEE", "Mean MEE STD"],
        index=[model_name]
    )

    return df