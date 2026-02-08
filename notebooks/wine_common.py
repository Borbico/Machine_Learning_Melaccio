import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial.distance import euclidean
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


columns = [
    "class",
    "a1", "a2", "a3", "a4", "a5", "a6",
    "id"
]

feature_names = columns[1:-1]
labels = [0, 1]

set_base_path = 'data/wine/winequality-{}.csv'


def _read_file(wine_path: str) -> DataFrame:
    """
    Read a file into a DataFrame
    :param monk_path: the monk file path
    :return: a DataFrame containing the original data
    """
    orig = pd.read_csv(wine_path, sep=r";")
    return orig


def _load_set() -> (DataFrame, DataFrame):
    """
    Load the monk training and testing sets by passing the monk set id, such as 1,2 or 3
    :return: the corresponding red and wine as DataFrames
    """
    red_path = set_base_path.format("red")
    white_path = set_base_path.format("white")

    # Load dataframe and add a specific
    # type column to each set
    df_red_orig = _read_file(red_path)
    df_red_orig['type'] = 1
    df_white_orig = _read_file(white_path)
    df_white_orig['type'] = 0

    return df_red_orig, df_white_orig


def stratified_split(class_col="type", test_size=0.2, seed=42, type: str = "join", quality_filter: list=None):
    """
    Stratified train/test split for Wine Quality dataset.

    Parameters
    ----------
    class_col : str
        Column used for stratification (default: 'type').
    test_size : float
        Fraction of data used for test set.
    seed : int
        Random seed for reproducibility.
    type : str
        Dataset selection:
        - 'join'  : red + white wines
        - 'red'   : only red wines
        - 'white' : only white wines

    Returns
    -------
    df_train : pd.DataFrame
        Training set.
    df_test : pd.DataFrame
        Test set.
    """

    df_red_orig, df_white_orig = _load_set()

    # Select dataset according to `type`
    if type == "join":
        df_full = pd.concat([df_red_orig, df_white_orig], ignore_index=True)

    elif type == "red":
        df_full = df_red_orig.copy()

    elif type == "white":
        df_full = df_white_orig.copy()

    else:
        raise ValueError("type must be one of: 'join', 'red', 'white'")

    if quality_filter is not None:
        df_full = df_full[~df_full["quality"].isin(quality_filter)]

    # If only one type is selected, stratification on `type` is meaningless
    if type in ("red", "white"):
        df_full = df_full.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_test = int(len(df_full) * test_size)
        df_test = df_full.iloc[:n_test].reset_index(drop=True)
        df_train = df_full.iloc[n_test:].reset_index(drop=True)

        return df_train, df_test

    # ---- Stratified split for joined dataset ----
    df_0 = df_full[df_full[class_col] == 0].copy()
    df_1 = df_full[df_full[class_col] == 1].copy()

    # Deterministic shuffle
    df_0 = df_0.sample(frac=1, random_state=seed)
    df_1 = df_1.sample(frac=1, random_state=seed)

    # Test sizes per class
    n0_test = int(len(df_0) * test_size)
    n1_test = int(len(df_1) * test_size)

    # Split
    df_test = pd.concat([
        df_0.iloc[:n0_test],
        df_1.iloc[:n1_test]
    ])

    df_train = pd.concat([
        df_0.iloc[n0_test:],
        df_1.iloc[n1_test:]
    ])

    # Final shuffle
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_test  = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_train, df_test


def split_and_prepare_dataset(df_train: DataFrame, df_test: DataFrame) -> (DataFrame, ndarray,DataFrame, ndarray, DataFrame, ndarray, DataFrame, ndarray):
    """
    Split dataset into training and testing sets eventually applying one-hot encoding
    :param df_orig: the original dataset
    :return: A tuple (X, y) representing the features and the labels
    """
    # The scaler
    scaler = MinMaxScaler()

    # Extract 'class' column which will be our labels
    y_tr = df_train["quality"].to_numpy()

    # Drop class and id columns to extract our set of features
    X_tr_unscaled = df_train.drop(columns=["quality","type"]) # The features
    # Transform features with MinMaxScaler
    X_tr = pd.DataFrame(
        scaler.fit_transform(X_tr_unscaled),
        columns=X_tr_unscaled.columns,
        index=X_tr_unscaled.index
    )

    # Extract 'class' column which will be our labels
    y_ts = df_test["quality"].to_numpy()

    # Drop class and id columns to extract our set of features
    X_ts_unscaled = df_test.drop(columns=["quality","type"]) # The features
    # Transform features with MinMaxScaler
    X_ts = pd.DataFrame(
        scaler.fit_transform(X_ts_unscaled),
        columns=X_ts_unscaled.columns,
        index=X_ts_unscaled.index
    )

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
            sorted(df_TR["quality"].unique().tolist()),
            _class_counts_str(df_TR)
        ],
        "Test": [
            df_TS.shape[0],
            df_TS.shape[1] - 2,
            sorted(df_TS["quality"].unique().tolist()),
            _class_counts_str(df_TS)
        ]
    })

    return dataset_overview
