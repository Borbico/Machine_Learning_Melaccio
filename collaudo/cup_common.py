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


def common_fold_strategy(n_split:int=5, shuffle:bool=True, random_state:int=42) -> KFold:
    """
    K-Fold cross-validation with shuffling is being used to obtain representative folds.
    A fixed random seed ensures reproducibility, and the same splitting strategy is applied to all models for fair comparison.
    :param n_split: The number of split
    :param shuffle: If the samples shave to shuffled.
    :param random_state:
    :return:
    """

    return KFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)


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


class FoldResults:

    def __init__(self):
        self._folds = []

    def append(self, fold_result: FoldResult):
        if not isinstance(fold_result, FoldResult):
            raise TypeError("Expected a FoldResult instance")
        self._folds.append(fold_result)

    def __iter__(self):
        return iter(self._folds)

    def __len__(self):
        return len(self._folds)

    def __getitem__(self, idx):
        return self._folds[idx]

    def get_fold(self, fold_nr: int) -> FoldResult:
        for f in self._folds:
            if f.fold_nr == fold_nr:
                return f
        raise KeyError(f"Fold {fold_nr} not found")

    def values(self, attr: str):
        """
        """
        vals = []
        for f in self._folds:
            v = getattr(f, attr)
            if v is not None:
                vals.append(v)
        return vals


class FoldResult:

    def __init__(self, fold_result):
        self._fold_nr = fold_result.get(FOLD_NR)

        self._epochs_tr_mse_mean = fold_result.get("epochs_tr_mse_mean")
        self._epochs_vl_mse_mean = fold_result.get("epochs_vl_mse_mean")
        self._epochs_tr_acc_mean = fold_result.get("epochs_tr_acc_mean")
        self._epochs_vl_acc_mean = fold_result.get("epochs_vl_acc_mean")

        self._epochs_tr_mae_mean = fold_result.get("epochs_tr_mae_mean")
        self._epochs_vl_mae_mean = fold_result.get("epochs_vl_mae_mean")

        self._epochs_tr_mee_mean = fold_result.get("epochs_tr_mee_mean")
        self._epochs_vl_mee_mean = fold_result.get("epochs_vl_mee_mean")

        self._fold_tr_mse = fold_result.get(FOLD_TR_MSE)
        self._fold_vl_mse = fold_result.get(FOLD_VL_MSE)
        self._fold_tr_acc = fold_result.get(FOLD_TR_ACC)
        self._fold_vl_acc = fold_result.get(FOLD_VL_ACC)
        self._fold_tr_mae = fold_result.get(FOLD_TR_MAE)
        self._fold_vl_mae = fold_result.get(FOLD_VL_MAE)
        self._fold_tr_mee = fold_result.get(FOLD_TR_MEE)
        self._fold_vl_mee = fold_result.get(FOLD_VL_MEE)

    @property
    def fold_nr(self):
        return self._fold_nr

    # -------- loss --------
    @property
    def epochs_tr_mse_mean(self):
        return self._epochs_tr_mse_mean

    @property
    def epochs_vl_mse_mean(self):
        return self._epochs_vl_mse_mean

    @property
    def fold_tr_mse(self):
        return self._fold_tr_mse

    @property
    def fold_vl_mse(self):
        return self._fold_vl_mse

    # -------- accuracy --------
    @property
    def epochs_tr_acc_mean(self):
        return self._epochs_tr_acc_mean

    @property
    def epochs_vl_acc_mean(self):
        return self._epochs_vl_acc_mean

    @property
    def fold_tr_acc(self):
        return self._fold_tr_acc

    @property
    def fold_vl_acc(self):
        return self._fold_vl_acc

    # -------- MAE --------
    @property
    def epochs_tr_mae_mean(self):
        return self._epochs_tr_mae_mean

    @property
    def epochs_vl_mae_mean(self):
        return self._epochs_vl_mae_mean

    @property
    def fold_tr_mae(self):
        return self._fold_tr_mae

    @property
    def fold_vl_mae(self):
        return self._fold_vl_mae

    # -------- MEE --------
    @property
    def epochs_tr_mee_mean(self):
        return self._epochs_tr_mee_mean

    @property
    def epochs_vl_mee_mean(self):
        return self._epochs_vl_mee_mean

    @property
    def fold_tr_mee(self):
        return self._fold_tr_mee

    @property
    def fold_vl_mee(self):
        return self._fold_vl_mee


def mee(y_true, y_pred):
    """
    MEE calc helper function returning np.mean
    :param y_true: the true label
    :param y_pred: the predicted label
    :return: float
    """
    return float(np.mean(np.linalg.norm(np.asarray(y_true) - np.asarray(y_pred), axis=1)))


def run_kfold(fold_model, X, y, folder_strategy) -> FoldResults:
    """
    Build a NN-like fold history dictionary for models without epochs (e.g., SVR, KNN).

    Produces the same keys style you use in NN:
    - tr_mee, tr_mee_std, vl_mee, vl_mee_std
    - tr_mae, tr_mae_std, vl_mae, vl_mae_std
    - tr_loss, vl_loss (here loss = MSE by default)
    - best_*, last_* are meaningful but trivial (best == last because 1 value per fold list).

    Notes:
    - Accuracy fields are set to None (not defined for regression).
    - If you want a different "loss", change loss_fn section.
    """

    Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    yn = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    # loop folds
    fold_results = FoldResults()
    for fold_nr, (tr_idx, vl_idx) in enumerate(folder_strategy.split(X)):

        X_tr, X_vl = Xn[tr_idx], Xn[vl_idx]
        y_tr, y_vl = yn[tr_idx], yn[vl_idx]

        print(f"  ")
        print(f"Perform fold: {fold_nr}, train size: {len(y_tr)}, val size: {len(vl_idx)}")

        fold_model = clone(fold_model)
        fold_model.fit(X_tr, y_tr)

        pred_tr = fold_model.predict(X_tr)
        pred_vl = fold_model.predict(X_vl)

        tr_mse = mean_squared_error(y_tr, pred_tr)
        tr_rmse = np.sqrt(tr_mse)
        vl_mse = mean_squared_error(y_vl, pred_vl)
        vl_rmse = np.sqrt(vl_mse)
        tr_mae = mean_absolute_error(y_tr, pred_tr)
        vl_mae = mean_absolute_error(y_vl, pred_vl)
        tr_mee = mee(y_tr, pred_tr)
        vl_mee = mee(y_vl, pred_vl)

        print(f"Fold summary")
        print("-" * 80)
        print(f"Validation MSE  | {vl_mse:.4f}")
        print(f"Validation RMSE | {vl_rmse:.4f}")
        print(f"Validation MEE  | {vl_mee:.4f}")
        print(f"Train           | samples: {len(y_tr)}")
        print(f"Validation      | samples: {len(y_vl)}")

        fold_results.append(FoldResult({
            "fold_nr": fold_nr,
            "tr_loss": tr_mse, "vl_loss": vl_mse,
            "tr_rmse": tr_rmse, "vl_rmse": vl_rmse,
            "tr_mae": tr_mae, "vl_mae": vl_mae,
            "tr_mee": tr_mee, "vl_mee": vl_mee,
            "best_tr_loss": tr_mse, "best_vl_loss": vl_mse,
            "best_tr_mee": tr_mee, "best_vl_mee": vl_mee
        }))

    return fold_results


def model_baseline(X, y, cv=common_fold_strategy(), baseline=DummyRegressor(strategy='mean'), scorer_name:str="mee"):

    scorer = common_scoring(scorer_name)

    baseline_scores = cross_val_score(baseline, X, y, cv=cv, scoring=scorer)

    print("-"*40)
    print(f"Baseline: {baseline}")
    print(f"Mean baseline: {-baseline_scores.mean()}")
    print(f"Scorer: {scorer}")
    print("Raw scores:", baseline_scores[:5])

    return -baseline_scores.mean()


def common_scoring(name:str= "mae"):
    """
    Return a scorer given o name:
        - mae: neg_mean_absolute_error
        - mse: neg_mean_squared_error
        - rmse: neg_root_mean_squared_error
        - med: neg_median_absolute_error
        - mee: mee_scorer()
    :param name: the name of the scorer
    :return: the scorer
    """

    if name == "mae":
        return "neg_mean_absolute_error"
    elif name == "mse":
        return "neg_mean_squared_error"
    elif name == "rmse":
        return "neg_root_mean_squared_error"
    elif name == "med":
        return "neg_median_absolute_error"
    elif name == "mee":
        return mee_scorer()
    else:
        raise ValueError("Unknown scorer")


def mee_scorer():
    mee.__name__ = "neg_mean_eclidean_error"
    ms = make_scorer(mee, greater_is_better=False)
    return ms


class SklearnRegressorRunner:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def new_model(self):
        return clone(self.base_estimator)

    def fit(self, model, X_tr, y_tr):
        model.fit(X_tr, y_tr)
        return model

    def predict(self, model, X):
        return model.predict(X)


class TorchNNRunner:
    """
    runner per la tua MLP:
    - model_template: un MLP NON addestrato (o un wrapper che contiene l'architettura)
    - train_fn(model, X_tr, y_tr, X_vl, y_vl) -> train_results (dict)
    - predict_fn(model, X) -> np.ndarray (N,K)
    """
    def __init__(self, model_template, train_fn, predict_fn):
        self.model_template = model_template
        self.train_fn = train_fn
        self.predict_fn = predict_fn

    def new_model(self):
        return copy.deepcopy(self.model_template)

    def fit(self, model, X_tr, y_tr, X_vl, y_vl):
        train_results = self.train_fn(model, X_tr, y_tr, X_vl, y_vl)
        return model, train_results

    def predict(self, model, X):
        return self.predict_fn(model, X)


def extract_best_pipeline_metrics_from_grid(gs):

    best_model = gs.best_estimator_.named_steps["regressor"]
    clean_params = {
        param.split("__")[-1]: value
        for param, value in gs.best_params_.items()
    }

    return best_model, clean_params


def grid_introspection(grid):

    print("Total model explored:", len(grid.cv_results_["params"]))
    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)
    print("Best model:", grid.best_estimator_)


def kfold(runner, X, y, folder_strategy, scaler=None) -> FoldResults:
    """
    Build a NN-like fold history dictionary for models without epochs (e.g., SVR, KNN).

    Produces the same keys style you use in NN:
    - tr_mee, tr_mee_std, vl_mee, vl_mee_std
    - tr_mae, tr_mae_std, vl_mae, vl_mae_std
    - tr_loss, vl_loss (here loss = MSE by default)
    - best_*, last_* are meaningful but trivial (best == last because 1 value per fold list).

    Notes:
    - Accuracy fields are set to None (not defined for regression).
    - If you want a different "loss", change loss_fn section.
    """

    Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    yn = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)


    # loop folds
    fold_results = FoldResults()
    for fold_nr, (tr_idx, vl_idx) in enumerate(folder_strategy.split(X)):

        #X_tr_raw, X_vl_raw = Xn[tr_idx], Xn[vl_idx]
        X_tr_raw, X_vl_raw = X.iloc[tr_idx], X.iloc[vl_idx]
        y_tr, y_vl = yn[tr_idx], yn[vl_idx]

        if scaler is not None:
            sc = clone(scaler)
            X_tr = sc.fit_transform(X_tr_raw)
            X_vl = sc.transform(X_vl_raw)
        else:
            X_tr = X_tr_raw
            X_vl = X_vl_raw

        print(f"  ")
        print(f"Perform fold: {fold_nr}, train size: {len(y_tr)}, val size: {len(vl_idx)}")

        unfitted_model = runner.new_model()
        model = runner.fit(unfitted_model, X_tr, y_tr)

        pred_tr = runner.predict(model, X_tr)
        pred_vl = runner.predict(model, X_vl)

        metrics = Metrics(y_tr, pred_tr, y_vl, pred_vl)
        tr_mse, vl_mse = metrics.tr_mse, metrics.vl_mse
        tr_rmse, vl_rmse = metrics.tr_rmse, metrics.vl_rmse
        tr_mae, vl_mae = metrics.tr_mae, metrics.vl_mae
        tr_mee, vl_mee = metrics.tr_mee, metrics.vl_mee

        print(f"Fold summary")
        print("-" * 80)
        print(f"Validation MSE  | {vl_mse:.4f}")
        print(f"Validation RMSE | {vl_rmse:.4f}")
        print(f"Validation MEE  | {vl_mee:.4f}")
        print(f"Validation MAE  | {vl_mae:.4f}")
        print(f"Train           | samples: {len(y_tr)}")
        print(f"Validation      | samples: {len(y_vl)}")

        fold_results.append(FoldResult({
            FOLD_NR: fold_nr,
            FOLD_TR_MSE: tr_mse, FOLD_VL_MSE: vl_mse,
            FOLD_TR_MEE: tr_mee, FOLD_VL_MEE: vl_mee,
            FOLD_TR_MAE: tr_mae, FOLD_VL_MAE: vl_mae
        }))

    return fold_results


class Metrics():

    def __init__(self, y_tr_true, y_tr_pred, y_vl_true, y_vl_pred):

        self._tr_mse = mean_squared_error(y_tr_true, y_tr_pred)
        self._tr_rmse = np.sqrt(self._tr_mse)
        self._vl_mse = mean_squared_error(y_vl_true, y_vl_pred)
        self._vl_rmse = np.sqrt(self._vl_mse)
        self._tr_mae = mean_absolute_error(y_tr_true, y_tr_pred)
        self._vl_mae = mean_absolute_error(y_vl_true, y_vl_pred)
        self._tr_mee = mee(y_tr_true, y_tr_pred)
        self._vl_mee = mee(y_vl_true, y_vl_pred)

    @property
    def tr_mse(self):
        return self._tr_mse

    @property
    def tr_rmse(self):
        return self._tr_rmse

    @property
    def vl_mse(self):
        return self._vl_mse

    @property
    def vl_rmse(self):
        return self._vl_rmse

    @property
    def tr_mae(self):
        return self._tr_mae

    @property
    def vl_mae(self):
        return self._vl_mae

    @property
    def tr_mee(self):
        return self._tr_mee

    @property
    def vl_mee(self):
        return self._vl_mee


def kfold_regression_table(fold_results: FoldResults, use: str="mee", derive_rmse=False):
    """
    Build a K-Fold summary table for regression.

    Parameters
    ----------
    kfold_results : list[dict]
        Output histories from training (one per fold).
    use : {"best", "last"}
        Whether to use best or last validation loss.
    derive_rmse : bool
        If True, derive RMSE from MSE (sqrt).

    Returns
    -------
    pandas.DataFrame
        Table with per-fold metrics + mean/std.
    """

    rows = []

    for i, fold_result in enumerate(fold_results, start=1):

        tr_value = getattr(fold_result,f'fold_tr_{use}')
        vl_value = getattr(fold_result,f'fold_vl_{use}')

        row = {
            "Fold": i,
            f'TR_{use.upper()}': tr_value,
            f'VL_{use.upper()}': vl_value,
            "Gap(TR-VL)": vl_value - tr_value
        }

        if derive_rmse:
            row["TR_RMSE"] = np.sqrt(tr_value)
            row["VL_RMSE"] = np.sqrt(vl_value)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Fold")

    # Add mean and std rows
    mean_row = df.mean()
    std_row = df.std()

    df.loc["MEAN"] = mean_row
    df.loc["STD"] = std_row

    #print(df)

    return df