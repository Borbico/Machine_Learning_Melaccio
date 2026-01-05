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
    # class column to each set
    df_red_orig = _read_file(red_path)
    df_red_orig['class'] = 1
    df_white_orig = _read_file(white_path)
    df_white_orig['class'] = 0

    return df_red_orig, df_white_orig


def stratified_split(class_col="class", test_size=0.2, seed=42):

    df_red_orig, df_white_orig = _load_set()
    df_full = pd.concat([df_red_orig, df_white_orig], ignore_index=True)

    rng = np.random.default_rng(seed)

    # separa per classe
    df_0 = df_full[df_full[class_col] == 0].copy()
    df_1 = df_full[df_full[class_col] == 1].copy()

    # shuffle deterministico
    df_0 = df_0.sample(frac=1, random_state=seed)
    df_1 = df_1.sample(frac=1, random_state=seed)

    # dimensioni test
    n0_test = int(len(df_0) * test_size)
    n1_test = int(len(df_1) * test_size)

    # split
    test_df = pd.concat([
        df_0.iloc[:n0_test],
        df_1.iloc[:n1_test]
    ])

    train_df = pd.concat([
        df_0.iloc[n0_test:],
        df_1.iloc[n1_test:]
    ])

    # rimescola train e test
    df_train = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_test  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_train, df_test


def split_and_prepare_dataset(df_train: DataFrame, df_test: DataFrame) -> (DataFrame, ndarray):
    """
    Split dataset into training and testing sets eventually applying one-hot encoding
    :param df_orig: the original dataset
    :return: A tuple (X, y) representing the features and the labels
    """
    # The scaler
    scaler = MinMaxScaler()

    # Extract 'class' column which will be our labels
    y_tr = df_train["class"].to_numpy()

    # Drop class and id columns to extract our set of features
    X_tr_unscaled = df_train.drop(columns=["class"]) # The features
    # Transform features with MinMaxScaler
    X_tr = pd.DataFrame(
        scaler.fit_transform(X_tr_unscaled),
        columns=X_tr_unscaled.columns,
        index=X_tr_unscaled.index
    )

    # Extract 'class' column which will be our labels
    y_ts = df_test["class"].to_numpy()

    # Drop class and id columns to extract our set of features
    X_ts_unscaled = df_test.drop(columns=["class"]) # The features
    # Transform features with MinMaxScaler
    X_ts = pd.DataFrame(
        scaler.fit_transform(X_ts_unscaled),
        columns=X_ts_unscaled.columns,
        index=X_ts_unscaled.index
    )

    return X_tr, y_tr, X_ts, y_ts


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
