
import pandas as pd
from numpy import ndarray
from pandas import DataFrame


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
