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







def split_and_prepare_dataset(df_train: DataFrame, ratio: float = 0.2, random_state:int = 42) -> (DataFrame, ndarray,DataFrame, ndarray, DataFrame, ndarray, DataFrame, ndarray):
    """
    Split dataset into training and testing sets eventually applying one-hot encoding
    :param df_orig: the original dataset
    :return: A tuple (X, y) representing the features and the labels
    """
    # The scaler
    scaler = MinMaxScaler()

    # divide dataframe in two df train and test based on ratio
    df_80, df_20 = train_test_split(
        df_train,
        test_size=ratio,
        random_state=random_state,
        shuffle=True
    )

    # Extract 'class' column which will be our labels
    y_tr = df_train[output_columns].to_numpy()

    # Drop class and id columns to extract our set of features
    X_tr_unscaled = df_80.drop(columns=["id"] + output_columns) # The features
    # Transform features with MinMaxScaler
    X_tr = pd.DataFrame(
        scaler.fit_transform(X_tr_unscaled),
        columns=X_tr_unscaled.columns,
        index=X_tr_unscaled.index
    )

    # Extract 'class' column which will be our labels
    y_tr = df_80[output_columns].to_numpy()

    # Drop class and id columns to extract our set of features
    X_ts_unscaled = df_20.drop(columns=["id"] + output_columns) # The features
    # Transform features with MinMaxScaler
    X_ts = pd.DataFrame(
        scaler.fit_transform(X_ts_unscaled),
        columns=X_ts_unscaled.columns,
        index=X_ts_unscaled.index
    )
    y_ts = df_20[output_columns].to_numpy()

    # Index realignment and reset
    X_tr = X_tr.reset_index(drop=True)
    #y_tr = y_tr.reset_index(drop=True)

    X_ts = X_ts.reset_index(drop=True)
    #y_ts = y_ts.reset_index(drop=True)

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
