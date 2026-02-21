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
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sympy.physics.control import Series

FOLD_NR = "fold_nr"
FOLD_TR_MSE = "fold_tr_mse"
FOLD_VL_MSE = "fold_vl_mse"
FOLD_TR_RMSE = "fold_tr_rmse"
FOLD_VL_RMSE = "fold_vl_rmse"
FOLD_TR_ACC = "fold_tr_acc"
FOLD_VL_ACC = "fold_vl_acc"
FOLD_TR_MAE = "fold_tr_mae"
FOLD_VL_MAE = "fold_vl_mae"
FOLD_TR_MEE = "fold_tr_mee"
FOLD_VL_MEE = "fold_vl_mee"

EPOCHS_VL_MEE_STD = "epochs_vl_mee_std"
EPOCHS_VL_MEE_MEAN = "epochs_vl_mee_mean"
EPOCHS_TR_MEE_STD = "epochs_tr_mee_std"
EPOCHS_TR_MEE_MEAN = "epochs_tr_mee_mean"
EPOCHS_VL_MAE_STD = "epochs_vl_mae_std"
EPOCHS_VL_MAE_MEAN = "epochs_vl_mae_mean"
EPOCHS_TR_MAE_STD = "epochs_tr_mae_std"
EPOCHS_TR_MAE_MEAN = "epochs_tr_mae_mean"
EPOCHS_VL_ACC_STD = "epochs_vl_acc_std"
EPOCHS_VL_ACC_MEAN = "epochs_vl_acc_mean"
EPOCHS_TR_ACC_STD = "epochs_tr_acc_std"
EPOCHS_TR_ACC_MEAN = "epochs_tr_acc_mean"
EPOCHS_VL_MSE_STD = "epochs_vl_mse_std"
EPOCHS_VL_MSE_MEAN = "epochs_vl_mse_mean"
EPOCHS_TR_MSE_STD = "epochs_tr_mse_std"
EPOCHS_TR_MSE_MEAN = "epochs_tr_mse_mean"

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


def dataset_introspection(df_TR: DataFrame, df_TS: DataFrame, class_introspection:bool=False) -> DataFrame:
    """
    Print a summary of the dataset introspection
    :param df_TR: the training dataframe
    :param df_TS: the testing dataframe
    :return: a summary of the dataset introspection in form of a DataFrame
    """

    if class_introspection:
        class_values = sorted(df_TR["quality"].unique().tolist()),
        class_balance = _class_counts_str(df_TR)
    else:
        class_values = []
        class_balance = []

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
            class_values,
            class_balance
        ],
        "Test": [
            df_TS.shape[0],
            df_TS.shape[1] - 2,
            [],#sorted(df_TS["quality"].unique().tolist()),
            [] #_class_counts_str(df_TS)
        ]
    })

    return dataset_overview


def common_fold_strategy(stratified:bool=False, n_split:int=5, shuffle:bool=True, random_state:int=42) -> KFold:
    """
    K-Fold cross-validation with shuffling is being used to obtain representative folds.
    A fixed random seed ensures reproducibility, and the same splitting strategy is applied to all models for fair comparison.
    :param n_split: The number of split
    :param shuffle: If the samples shave to shuffled.
    :param random_state:
    :return:
    """
    if stratified:
        return StratifiedKFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    else:
        return KFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)


# def plot_kfold_mee(kf_result):
#     plot_kfold_bar(extract_fold_history(kf_result, "fold_vl_mee"), "Validation MEE", "KFold Validation MEE per fold")


def plot_kfold_metric(results: [tuple], key:str, subkey:str="vl"):

    data = []
    key_composed = map_key_to_metric(key, subkey)
    for result in results:
        values = extract_fold_history(result[0], key_composed)
        data.append((values, result[1]))
    plot_kfold_bars(data, f"Validation {key.upper()}", f"KFold Validation {key.upper()} per fold")


def plot_kfold_mse(non_nested_result, nested_result):

    plot_kfold_bars([
        (extract_fold_history(non_nested_result, FOLD_VL_MSE), "non-nested"),
        (extract_fold_history(nested_result, FOLD_VL_MSE), "nested")
    ], "Validation MSE", "KFold Validation MSE per fold")


def plot_kfold_rmse(results: [tuple]):
    data = []
    for result in results:
        data.append((extract_fold_history(result[0], FOLD_VL_MSE), result[1]))
    plot_kfold_bars(data, "Validation RMSE", "KFold Validation RMSE per fold")


def plot_kfold_mae(non_nested_result, nested_result):

    plot_kfold_bars([
        (extract_fold_history(non_nested_result, FOLD_VL_MAE), "Non-nested"),
        (extract_fold_history(nested_result, FOLD_VL_MAE), "Nested")
    ], "Validation MAE", "KFold Validation MAE per fold")


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


def plot_kfold_bars(histories,ylabel: str, title: str):
    """
    Plot multiple fold histories side-by-side.

    histories: list of tuples
        [(history_values, label), ...]
    """

    sns.set_theme(style="whitegrid")

    n_methods = len(histories)
    vals_list = [np.array(h[0], dtype=float) for h in histories]
    labels = [h[1] for h in histories]

    n_folds = len(vals_list[0])
    folds = np.arange(1, n_folds + 1)

    x = np.arange(n_folds)
    total_width = 0.8
    width = total_width / n_methods

    plt.figure()

    for i, (vals, label) in enumerate(zip(vals_list, labels)):
        offset = (i - (n_methods - 1) / 2) * width
        mean_val = vals.mean()
        std_val = vals.std()

        # Bars
        plt.bar(x + offset, vals, width, label=label)

        # Error bars (std)
        plt.errorbar(x + offset, vals, yerr=std_val, fmt="none", ecolor="black", capsize=4, linewidth=1)

        # Mean line
        plt.axhline(mean_val,linestyle="--",linewidth=1.5,alpha=0.6,label=f"{label} Mean = {mean_val:.4f}")

    plt.xticks(x, folds)
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

        self._epochs_tr_mse_mean = fold_result.get(EPOCHS_TR_MSE_MEAN)
        self._epochs_vl_mse_mean = fold_result.get(EPOCHS_VL_MSE_MEAN)
        self._epochs_tr_acc_mean = fold_result.get(EPOCHS_TR_ACC_MEAN)
        self._epochs_vl_acc_mean = fold_result.get(EPOCHS_VL_ACC_MEAN)

        self._epochs_tr_mae_mean = fold_result.get(EPOCHS_TR_MAE_MEAN)
        self._epochs_vl_mae_mean = fold_result.get(EPOCHS_VL_MAE_MEAN)

        self._epochs_tr_mee_mean = fold_result.get(EPOCHS_TR_MEE_MEAN)
        self._epochs_vl_mee_mean = fold_result.get(EPOCHS_VL_MEE_MEAN)

        self._fold_tr_mse = fold_result.get(FOLD_TR_MSE)
        self._fold_vl_mse = fold_result.get(FOLD_VL_MSE)
        self._fold_tr_rmse = fold_result.get(FOLD_TR_RMSE)
        self._fold_vl_rmse = fold_result.get(FOLD_VL_RMSE)
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

    @property
    def fold_tr_rmse(self):
        return self._fold_tr_rmse

    @property
    def fold_vl_rmse(self):
        return self._fold_vl_rmse

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


def model_baseline(X, y, cv=common_fold_strategy(), baseline=DummyRegressor(strategy='mean'), scorer_name:str="mee"):
    """
    Define a score based on a base model
    :param X: The train data
    :param y: The label
    :param cv: fold strategy
    :param baseline: the model
    :param scorer_name: the scorer name as in common_scoring(scorer_name)
    :return:
    """

    scorer = common_scoring(scorer_name)

    baseline_scores = cross_val_score(baseline, X, y, cv=cv, scoring=scorer)

    print("-"*40)
    print(f"Baseline: {baseline}")
    print(f"Mean baseline: {-baseline_scores.mean()}")
    print(f"Scorer: {scorer}")
    print("Raw scores:", baseline_scores[:5])

    return -baseline_scores.mean()


def map_key_to_metric(key: str, subkey:str="vl"):

    if key == "mae":
        return f"fold_{subkey}_mae"
    elif key == "mse":
        return f"fold_{subkey}_mse"
    elif key == "rmse":
        return f"fold_{subkey}_rmse"
    elif key == "mee":
        return f"fold_{subkey}_mee"

    raise NameError


def common_scoring(name:str= "mae"):
    """
    Return a scorer given a name:
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
    elif name == "acc":
        return "accuracy"
    elif name == "bacc":
        return "balanced_accuracy"
    else:
        raise ValueError("Unknown scorer")


def mee_scorer():
    mee.__name__ = "neg_mean_euclidean_error"
    ms = make_scorer(mee, greater_is_better=False)
    return ms


def assess_cv_robustness(cv, model, X, y, scoring:str, seeds:list=[10, 20, 30, 40, 50]):
    """
    Perform a cross_val_score in order to compare model score with cv seeds
    :param cv:
    :param model:
    :param X:
    :param y:
    :param scoring:
    :param seeds:
    :return:
    """

    params = cv.__dict__.copy()
    for seed in seeds:
        params["random_state"] = seed
        cv_copy = cv.__class__(**params)
        scores = cross_val_score(
            model,
            X, y,
            cv=cv_copy,
            scoring=scoring,
        )

        print('seed: {} - mean {:.3f} ± {:.3f}'.format(seed, scores.mean(), scores.std()))


def plot_top_models(results, scoring_name:str, label_row:Series, TOP_N=10):
    res = pd.DataFrame(results)

    # Sort: best -> worst
    res_sorted = res.sort_values(["mean_test_score", "rank_test_score"], ascending=[False, True]).reset_index(drop=True)

    # Top e Worst
    top = res_sorted.head(TOP_N).copy()
    worst = res_sorted.tail(TOP_N).copy().iloc[::-1].copy()  # inverti: peggiore in alto

    # Etichette (richiede che label_row esista già)
    top["model_label"] = top.apply(label_row, axis=1)
    worst["model_label"] = worst.apply(label_row, axis=1)

    # --- Plot TOP ---
    plt.figure(figsize=(9, 3))
    scores = top["mean_test_score"].astype(float).values
    errs = top["std_test_score"].astype(float).values
    plt.barh(top["model_label"], scores, xerr=errs)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Mean CV score ({scoring_name})")
    plt.title(f"Top {TOP_N} models")
    plt.tight_layout()
    plt.show()

    # --- Plot WORST ---
    plt.figure(figsize=(9, 3))
    scores = worst["mean_test_score"].astype(float).values
    errs = worst["std_test_score"].astype(float).values
    plt.barh(worst["model_label"], scores, xerr=errs)
    plt.gca().invert_yaxis()  # peggiore in alto
    plt.xlabel(f"Mean CV score ({scoring_name})")
    plt.title(f"Worst {TOP_N} models from GridSearchCV")
    plt.tight_layout()
    plt.show()


class SklearnNestedRegressorRunner:
    def __init__(self, grid):
        self.base_grid = grid

    def new_model(self):
        return clone(self.base_grid)

    def fit(self, gs, X_tr, y_tr):
        gs.fit(X_tr, y_tr)
        grid_introspection(gs)
        return gs
        #return gs.best_estimator_

    def predict(self, model, X):
        return model.predict(X)


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


def extract_best_pipeline_metrics_from_grid(gs, step_name:str="model"):

    best_model = gs.best_estimator_.named_steps[step_name]
    clean_params = {
        param.split("__")[-1]: value
        for param, value in gs.best_params_.items()
    }

    return best_model, clean_params


def grid_introspection(grid):

    print("Grid summary")
    print("-"*40)
    print("Total model explored:", len(grid.cv_results_["params"]), " splits:", grid.n_splits_ )
    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)
    print("Best model:", grid.best_estimator_)


def kfold(runner, X, y, folder_strategy) -> FoldResults:

    Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    yn = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    # loop folds
    fold_results = FoldResults()
    for fold_nr, (tr_idx, vl_idx) in enumerate(folder_strategy.split(X), start=1):

        X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
        y_tr, y_vl = yn[tr_idx], yn[vl_idx]

        print(f"  ")
        print(f"Perform fold: {fold_nr}, train size: {len(y_tr)}, val size: {len(vl_idx)}")

        unfitted_model = runner.new_model()
        model = runner.fit(unfitted_model, X_tr, y_tr)

        pred_tr = runner.predict(model, X_tr)
        pred_vl = runner.predict(model, X_vl)

        metrics = MetricsPair(y_tr, pred_tr, y_vl, pred_vl)
        tr_mse, vl_mse = metrics.tr_mse, metrics.vl_mse
        tr_rmse, vl_rmse = metrics.tr_rmse, metrics.vl_rmse
        tr_mae, vl_mae = metrics.tr_mae, metrics.vl_mae
        tr_mee, vl_mee = metrics.tr_mee, metrics.vl_mee

        print(f" ")
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
            FOLD_TR_MAE: tr_mae, FOLD_VL_MAE: vl_mae,
            FOLD_TR_RMSE: np.sqrt(tr_mae), FOLD_VL_RMSE: np.sqrt(vl_mae)
        }))

    return fold_results


# def nested_cv(runner, X, y, outer_cv, grid, scaler=None, verbose=True) -> FoldResults:
#     """
#     Nested CV returning the same FoldResults structure as kfold():
#       - Outer CV: model assessment (this is what you plot)
#       - Inner CV: GridSearchCV hyperparameter selection
#     """
#
#     yn = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
#
#     fold_results = FoldResults()
#
#     # OUTER loop
#     for fold_nr, (tr_idx, vl_idx) in enumerate(outer_cv.split(X), start=1):
#
#         # keep pandas to preserve feature names
#         X_tr_raw, X_vl_raw = X.iloc[tr_idx], X.iloc[vl_idx]
#         y_tr, y_vl = yn[tr_idx], yn[vl_idx]
#
#         if scaler is not None:
#             sc = clone(scaler)
#             X_tr = sc.fit_transform(X_tr_raw)
#             X_vl = sc.transform(X_vl_raw)
#         else:
#             X_tr, X_vl = X_tr_raw, X_vl_raw
#
#         if verbose:
#             print(f"  ")
#             print(f"Perform (outer) fold: {fold_nr}, train size: {len(y_tr)}, val size: {len(vl_idx)}")
#
#         # INNER grid search on outer-train only
#         base_estimator = runner.new_model()          # unfitted pipeline/model template
#         gs = clone(grid)
#         gs.fit(X_tr, y_tr)
#         best_model = gs.best_estimator_
#
#         # unfitted_model = runner.new_model()
#         # model = runner.fit(unfitted_model, X_tr, y_tr)
#
#         # class SklearnNestedRegressorRunner:
#         #     def __init__(self, grid):
#         #         self.base_grid = grid
#         #
#         #     def new_model(self):
#         #         return clone(self.grid)
#         #
#         #     def fit(self, gs, X_tr, y_tr):
#         #         gs.fit(X_tr, y_tr)
#         #         return gs.best_estimator_
#         #
#         #     def predict(self, model, X):
#         #         return model.predict(X)
#
#         # evaluate best_model on OUTER train/val
#         pred_tr = runner.predict(best_model, X_tr)
#         pred_vl = runner.predict(best_model, X_vl)
#
#         metrics = Metrics(y_tr, pred_tr, y_vl, pred_vl)
#
#         if verbose:
#             print(f"Inner fold summary")
#             print("-" * 80)
#             print(f"Validation MSE  | {metrics.vl_mse:.4f}")
#             print(f"Validation RMSE | {metrics.vl_rmse:.4f}")
#             print(f"Validation MEE  | {metrics.vl_mee:.4f}")
#             print(f"Validation MAE  | {metrics.vl_mae:.4f}")
#             print(f"Train           | samples: {len(y_tr)}")
#             print(f"Validation      | samples: {len(y_vl)}")
#             print("-" * 80)
#             grid_introspection(gs)
#
#         fold_results.append(FoldResult({
#             FOLD_NR: fold_nr,
#             FOLD_TR_MSE: metrics.tr_mse, FOLD_VL_MSE: metrics.vl_mse,
#             FOLD_TR_MEE: metrics.tr_mee, FOLD_VL_MEE: metrics.vl_mee,
#             FOLD_TR_MAE: metrics.tr_mae, FOLD_VL_MAE: metrics.vl_mae,
#
#             "best_params": gs.best_params_,
#         }))
#
#     return fold_results

class Metrics():

    def __init__(self, y_true, y_pred):
        self._mse = mean_squared_error(y_true, y_pred)
        self._rmse = np.sqrt(self._mse)
        self._mae = mean_absolute_error(y_true, y_pred)
        self._mee = mee(y_true, y_pred)

    @property
    def mse(self):
        return self._mse

    @property
    def rmse(self):
        return self._rmse

    @property
    def mae(self):
        return self._mae

    @property
    def mee(self):
        return self._mee


class MetricsPair():

    def __init__(self, y_tr_true, y_tr_pred, y_vl_true, y_vl_pred):

        tr_metrics = Metrics(y_tr_true, y_tr_pred)
        vl_metrics = Metrics(y_vl_true, y_vl_pred)
        self._tr_mse = tr_metrics.mse
        self._tr_rmse = tr_metrics.rmse
        self._vl_mse = vl_metrics.mse
        self._vl_rmse = vl_metrics.rmse
        self._tr_mae = tr_metrics.mae
        self._vl_mae = vl_metrics.mae
        self._tr_mee = tr_metrics.mee
        self._vl_mee = vl_metrics.mee

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


def assess_sklearn_cv_robustness(cv, model, X, y, scoring="accuracy", seeds:list=[10, 20, 30, 40, 50]):

    params = cv.__dict__.copy()
    for seed in seeds:
        params["random_state"] = seed
        cv_copy = cv.__class__(**params)
        scores = cross_val_score(
            model,
            X, y,
            cv=cv_copy,
            scoring=scoring,
        )

        print('seed: {} - mean {:.3f} ± {:.3f}'.format(seed, scores.mean(), scores.std()))


def plot_learning_curve(model, X, y, cv, ax=None, score_correction:int=1, scoring: str = "accuracy"):
    """
    Plot a learning curve for a given model (classifier or regressor).
    If scoring is a neg_* loss, the curve is plotted as the corresponding positive loss.
    """

    random_state = getattr(cv, "random_state", None)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        train_sizes = np.linspace(0.2, 1.0, 15),
        shuffle=True,
        random_state=random_state
    )

    train_mean = score_correction *  train_scores.mean(axis=1)
    val_mean = score_correction * val_scores.mean(axis=1)

    # Decide label + possible sign flip
    y_label = "Score"
    train_label = "Training score"
    val_label = "CV score"
    title_metric = scoring

    if isinstance(scoring, str) and scoring.startswith("neg_"):
        # Convert from negative score to positive loss for readability
        train_mean = -train_mean
        val_mean = -val_mean

        if scoring == "neg_mean_absolute_error":
            y_label = "MAE"
            title_metric = "MAE"
        elif scoring == "neg_mean_squared_error":
            y_label = "MSE"
            title_metric = "MSE"
        elif scoring == "neg_root_mean_squared_error":
            y_label = "RMSE"
            title_metric = "RMSE"
        else:
            y_label = scoring.replace("neg_", "").replace("_", " ").upper()
            title_metric = y_label

        train_label = f"Training {y_label}"
        val_label = f"CV {y_label}"
    elif scoring == "accuracy":
        y_label = "Accuracy"
        train_label = "Training accuracy"
        val_label = "CV accuracy"
        title_metric = "Accuracy"

    # plot
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(train_sizes_abs, train_mean, label=train_label)
    ax.plot(train_sizes_abs, val_mean, label=val_label)

    ax.set_xlabel("Training set size")
    ax.set_ylabel(y_label)
    ax.set_title(f"Learning curve")
    ax.legend()
    ax.grid(True)
