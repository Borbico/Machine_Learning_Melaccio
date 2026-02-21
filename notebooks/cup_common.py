import copy

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial.distance import euclidean
from sklearn import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

FOLD_NR = "fold_nr"
FOLD_TR_MSE = "fold_tr_mse"
FOLD_VL_MSE = "fold_vl_mse"
FOLD_TR_ACC = "fold_tr_acc"
FOLD_VL_ACC = "fold_vl_acc"
FOLD_TR_MAE = "fold_tr_mae"
FOLD_VL_MAE = "fold_vl_mae"
FOLD_TR_MEE = "fold_tr_mee"
FOLD_VL_MEE = "fold_vl_mee"

output_columns = ["t1", "t2", "t3", "t4"]
base_columns = [
    "id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"
]

tr_columns = base_columns + output_columns

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


def prepare_dataset_for_hold_out(df: DataFrame, ratio: float = 0.2, random_state:int = 42, scaler=None) -> (DataFrame, ndarray, DataFrame, ndarray, DataFrame, ndarray, DataFrame, ndarray):
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
        df,
        test_size=ratio,
        random_state=random_state,
        shuffle=True
    )

    X_tr, y_tr = prepare_dataset(df_80, scaler=scaler, fit_scaler=True)
    X_ts, y_ts = prepare_dataset(df_20, scaler=scaler, fit_scaler=False)

    return X_tr, y_tr, X_ts, y_ts


def plot_kfold_mee(kf_result):
    plot_kfold_bar(extract_fold_history(kf_result, "fold_vl_mee"), "Validation MEE", "KFold Validation MEE per fold")


def plot_kfold_mse(kf_result):
    plot_kfold_bar(extract_fold_history(kf_result, "fold_vl_mse"), "Validation MSE", "KFold Validation MSE per fold")


def plot_kfold_rmse(kf_result):
    mse = extract_fold_history(kf_result, "fold_vl_mse")
    rmse = np.sqrt(np.array(mse, dtype=float))
    plot_kfold_bar(rmse, "Validation RMSE", "KFold Validation RMSE per fold")


def plot_kfold_mae(kf_result):
    plot_kfold_bar(extract_fold_history(kf_result, "fold_vl_mae"), "Validation MAE", "KFold Validation MAE per fold")


def plot_kfold_acc(kf_result):
    plot_kfold_bar(extract_fold_history(kf_result, "fold_vl_acc"), "Validation ACC", "KFold Validation ACC (best) per fold")


def plot_kfold_bar(history, ylabel:str, title:str):
    """
    Bar plot helper for validation metric per fold + mean line (seaborn version).
    :param history: the data to be plotted
    :param ylabel: the label of the y axis
    :param title: the title of the y axis
    """

    vals = np.array(history, dtype=float)

    df = pd.DataFrame({
        "Fold": np.arange(1, len(vals) + 1),
        ylabel: vals
    })

    mean_val = vals.mean()
    std_val = vals.std()

    sns.set_theme(style="whitegrid")

    plt.figure()
    ax = sns.barplot(data=df,x="Fold",y=ylabel,errorbar=None)

    # (± std)
    ax.errorbar(x=np.arange(len(vals)),y=vals,yerr=std_val,fmt="none",ecolor="black",capsize=5,linewidth=1)

    # Mean
    plt.axhline(mean_val,linestyle="--",linewidth=2,color="black",label=f"Mean = {mean_val:.4f}")

    plt.xlabel("Fold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def extract_fold_history(fold_histories, key):
    return np.array([getattr(fold_history, key) for fold_history in fold_histories])


def mean_std_from_kfold(vals):

    mean_m = np.mean(vals)
    std_m = np.std(vals)
    return mean_m, std_m


def extract_mean_std(fold_histories, key:str):

    best_vals = extract_fold_history(fold_histories, key)
    return mean_std_from_kfold(best_vals)


# class FoldResults:
#
#     def __init__(self):
#         self._folds = []
#
#     def append(self, fold_result: FoldResult):
#         if not isinstance(fold_result, FoldResult):
#             raise TypeError("Expected a FoldResult instance")
#         self._folds.append(fold_result)
#
#     def __iter__(self):
#         return iter(self._folds)
#
#     def __len__(self):
#         return len(self._folds)
#
#     def __getitem__(self, idx):
#         return self._folds[idx]
#
#     def get_fold(self, fold_nr: int) -> FoldResult:
#         for f in self._folds:
#             if f.fold_nr == fold_nr:
#                 return f
#         raise KeyError(f"Fold {fold_nr} not found")
#
#     def values(self, attr: str):
#         """
#         """
#         vals = []
#         for f in self._folds:
#             v = getattr(f, attr)
#             if v is not None:
#                 vals.append(v)
#         return vals


# class FoldResult:
#
#     def __init__(self, fold_result):
#         self._fold_nr = fold_result.get(FOLD_NR)
#
#         self._epochs_tr_mse_mean = fold_result.get("epochs_tr_mse_mean")
#         self._epochs_vl_mse_mean = fold_result.get("epochs_vl_mse_mean")
#         self._epochs_tr_acc_mean = fold_result.get("epochs_tr_acc_mean")
#         self._epochs_vl_acc_mean = fold_result.get("epochs_vl_acc_mean")
#
#         self._epochs_tr_mae_mean = fold_result.get("epochs_tr_mae_mean")
#         self._epochs_vl_mae_mean = fold_result.get("epochs_vl_mae_mean")
#
#         self._epochs_tr_mee_mean = fold_result.get("epochs_tr_mee_mean")
#         self._epochs_vl_mee_mean = fold_result.get("epochs_vl_mee_mean")
#
#         self._fold_tr_mse = fold_result.get(FOLD_TR_MSE)
#         self._fold_vl_mse = fold_result.get(FOLD_VL_MSE)
#         self._fold_tr_acc = fold_result.get(FOLD_TR_ACC)
#         self._fold_vl_acc = fold_result.get(FOLD_VL_ACC)
#         self._fold_tr_mae = fold_result.get(FOLD_TR_MAE)
#         self._fold_vl_mae = fold_result.get(FOLD_VL_MAE)
#         self._fold_tr_mee = fold_result.get(FOLD_TR_MEE)
#         self._fold_vl_mee = fold_result.get(FOLD_VL_MEE)
#
#     @property
#     def fold_nr(self):
#         return self._fold_nr
#
#     # -------- loss --------
#     @property
#     def epochs_tr_mse_mean(self):
#         return self._epochs_tr_mse_mean
#
#     @property
#     def epochs_vl_mse_mean(self):
#         return self._epochs_vl_mse_mean
#
#     @property
#     def fold_tr_mse(self):
#         return self._fold_tr_mse
#
#     @property
#     def fold_vl_mse(self):
#         return self._fold_vl_mse
#
#     # -------- accuracy --------
#     @property
#     def epochs_tr_acc_mean(self):
#         return self._epochs_tr_acc_mean
#
#     @property
#     def epochs_vl_acc_mean(self):
#         return self._epochs_vl_acc_mean
#
#     @property
#     def fold_tr_acc(self):
#         return self._fold_tr_acc
#
#     @property
#     def fold_vl_acc(self):
#         return self._fold_vl_acc
#
#     # -------- MAE --------
#     @property
#     def epochs_tr_mae_mean(self):
#         return self._epochs_tr_mae_mean
#
#     @property
#     def epochs_vl_mae_mean(self):
#         return self._epochs_vl_mae_mean
#
#     @property
#     def fold_tr_mae(self):
#         return self._fold_tr_mae
#
#     @property
#     def fold_vl_mae(self):
#         return self._fold_vl_mae
#
#     # -------- MEE --------
#     @property
#     def epochs_tr_mee_mean(self):
#         return self._epochs_tr_mee_mean
#
#     @property
#     def epochs_vl_mee_mean(self):
#         return self._epochs_vl_mee_mean
#
#     @property
#     def fold_tr_mee(self):
#         return self._fold_tr_mee
#
#     @property
#     def fold_vl_mee(self):
#         return self._fold_vl_mee
