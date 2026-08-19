import copy
from typing import Callable, cast

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, KFold, GridSearchCV
import torch.nn.functional as F
from pandas import Series


RANDOM_STATE = 42
SEED = 1

MEE = "mee"
MAE = "mae"
MSE = "mse"
RMSE = "rmse"
R2 = "r2"
ACC = "acc"
MED = "med"
BAC = "bac"
LOSS = "loss"
CUSTOM = "custom"

_BASE = "fold"
_TR, _VL = "tr", "vl"
FOLD_NR = f"{_BASE}_nr"
FOLD_BEST_EPOCH = f"{_BASE}_best_epoch"
FOLD_TR_MSE = f"fold_tr_{MSE}"
FOLD_VL_MSE = f"fold_vl_{MSE}"
FOLD_TR_RMSE = f"fold_tr_{RMSE}"
FOLD_VL_RMSE = f"fold_vl_{RMSE}"
FOLD_TR_ACC = f"fold_tr_{ACC}"
FOLD_VL_ACC = f"fold_vl_{ACC}"
FOLD_TR_MAE = f"fold_tr_{MAE}"
FOLD_VL_MAE = f"fold_vl_{MAE}"
FOLD_TR_MEE = f"fold_tr_{MEE}"
FOLD_VL_MEE = f"fold_vl_{MEE}"
FOLD_TR_R2 = f"fold_tr_{R2}"
FOLD_VL_R2 = f"fold_vl_{R2}"
FOLD_VL_LOSS = f"fold_vl_{LOSS}"
FOLD_TR_LOSS = f"fold_tr_{LOSS}"
FOLD_TR_CUSTOM = f"fold_tr_{CUSTOM}"
FOLD_VL_CUSTOM = f"fold_vl_{CUSTOM}"

EPOCHS_TR_MSE = "epochs_tr_mse"
EPOCHS_VL_MSE = "epochs_vl_mse"
EPOCHS_TR_RMSE = "epochs_tr_rmse"
EPOCHS_VL_RMSE = "epochs_vl_rmse"
EPOCHS_TR_ACC = "epochs_tr_acc"
EPOCHS_VL_ACC = "epochs_vl_acc"
EPOCHS_TR_MAE = "epochs_tr_mae"
EPOCHS_VL_MAE = "epochs_vl_mae"
EPOCHS_TR_MEE = "epochs_tr_mee"
EPOCHS_VL_MEE = "epochs_vl_mee"
EPOCHS_TR_CUSTOM = "epochs_tr_custom"
EPOCHS_VL_CUSTOM = "epochs_vl_custom"
EPOCHS_VL_LOSS = f"epochs_vl_{LOSS}"
EPOCHS_TR_LOSS = f"epochs_tr_{LOSS}"


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
base_columns = ["id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

tr_columns = base_columns + output_columns

set_base_path = 'data/cup/ML-CUP25-{}.csv'


def sort_with_loss_first(losses: dict):
    """
    Reorganize losses dictionary by putting LOSS key first
    :param losses:
    :return:
    """
    if "LOSS" in losses:
        return {"LOSS": losses["LOSS"], **{k: v for k, v in losses.items() if k != "LOSS"}}
    return losses


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


def plot_kfold_metric(results: list[tuple[FoldResults,str]], key:str, subkey:str="vl"):

    data = []
    key_composed = map_key_to_metric(key, subkey)
    for result in results:
        values = extract_fold_history(result[0], key_composed)
        data.append((values, result[1]))
    plot_kfold_bars(data, f"Validation {key.upper()}", f"KFold Validation {key.upper()} per fold")


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


def mean_std_from_kfold(vals:list(float)) -> tuple(float,float):
    """
    Compute the mean and std value of a series of float values.
    :param vals: a list of float values
    :return: a tuple of meand and std
    """

    mean_m = np.mean(vals)
    std_m = np.std(vals)
    return mean_m, std_m


def extract_mean_std(fold_histories: dict, key:str) -> tuple(float,float):
    """
    Extract mean and standard deviation from kfold history.
    :param fold_histories: kfold history
    :param key: the metric to extract
    :return: a tuple of meand and std
    """

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

    def extract(self,key: str, subkey: str):

        data = []
        key_composed = map_key_to_metric(key, subkey)
        for result in self.values(key_composed):
            values = extract_fold_history(result[0], key_composed)
            data.append((values, result[1]))


class FoldResult:

    def __init__(self):
        """
        Holds a kfold cross validation results
        """
        self._data = dict()
        return

    def __getattr__(self, name: str):
        """
        Allows attribute-style access:
        fr.fold_vl_mse
        """
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"There is no attribute '{name}'")

    def set_metric(self, name: str, value):
        """
        Set a single metric.
        :param name: metric name (e.g., 'fold_vl_mse')
        :param value: metric value
        """
        self._data[name] = value


def mee(y_true:ndarray, y_pred:ndarray) -> float:
    """
    MEE calc helper function returning np.mean
    :param y_true: the true label
    :param y_pred: the predicted label
    :return: float(np.mean(np.linalg.norm(np.asarray(y_true) - np.asarray(y_pred), axis=1)))
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

    return f"fold_{subkey}_{key}"


class Scoring:

    def __init__(self, name: str):

        if name == MAE:
            self._scoring_fn = mae
            self._grid_scoring = self._scoring_name = "neg_mean_absolute_error"
            self._reducer = min
        elif name == MSE:
            self._scoring_fn = mse
            self._reducer = min
            self._grid_scoring = self._scoring_name = "neg_mean_squared_error"
        elif name == RMSE:
            self._scoring_fn = rmse
            self._reducer = min
            self._grid_scoring = self._scoring_name = "neg_root_mean_squared_error"
        # elif name == MED: not implemented yet
        #     self._fn = med
        #     self._scoring = "neg_median_absolute_error"
        elif name == MEE:
            self._scoring_fn = mee_scorer
            self._reducer = min
            self._grid_scoring = mee_scorer()
            self._scoring_name = "neg_mean_euclidean_error"
        elif name == ACC:
            self._scoring_fn = None
            self._reducer = max
            self._grid_scoring = self._scoring_name = "accuracy"
        elif name == BAC:
            self._scoring_fn = None
            self._reducer = max
            self._grid_scoring = self._scoring_name = "balanced_accuracy"
        else:
            raise ValueError("Unknown scorer")

    @property
    def scoring_fn(self):
        return self._scoring_fn

    @property
    def grid_scoring(self):
        return self._grid_scoring


class ScoringFactory:
    def __init__(self, scores_dict: dict):
        return


def common_scoring(name:str= "mae") -> str:
    """
    Return a scorer identifier given a name:
        - mae: neg_mean_absolute_error
        - mse: neg_mean_squared_error
        - rmse: neg_root_mean_squared_error
        - med: neg_median_absolute_error
        - mee: mee_scorer()
        - acc: accuracy
        - bac: balanced_accuracy
    :param name: the name of the scorer
    :return: the scorer full name
    """

    if name == MAE:
        return "neg_mean_absolute_error"
    elif name == MSE:
        return "neg_mean_squared_error"
    elif name == RMSE:
        return "neg_root_mean_squared_error"
    elif name == MED:
        return "neg_median_absolute_error"
    elif name == MEE:
        return mee_scorer()
    elif name == ACC:
        return "accuracy"
    elif name == BAC:
        return "balanced_accuracy"
    else:
        raise ValueError("Unknown scorer")


def mee_scorer(greater_is_better:bool=False) -> Callable[[BaseEstimator, np.ndarray, np.ndarray], float]:
    """
    The mee scorer to be used in GridSearchCV
    :return: scorer
    """
    mee.__name__ = "neg_mean_euclidean_error"
    return make_scorer(mee, greater_is_better=greater_is_better)


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
    """
    To be used in nested CV
    """
    def __init__(self, grid: GridSearchCV):
        """
        Utility class for wrapping a GridSearchCV that needs to be instantiated and fitted
        from scratch and then being used in kfold or predictions.
        Steps are:
            - runner = SklearnNestedRegressorRunner(grid)
            - model = runner.new_model()
            - runner.fit(model, X, y)
            - runner.predict(model, X)
        The grid is cloned internally both in 'init' and at each new_model invocation to avoid unintentional fitting or modification
        of the model made externally.
        :param grid: the grid search to be used as a template
        """
        self.base_grid = clone(grid)

    def unfitted_model(self) -> GridSearchCV:
        """
        Return a cloned unfitted GridSearchCV model
        :return: GridSearchCV
        """
        return clone(self.base_grid)

    def fitted_model(self, X_tr: DataFrame, y_tr: ndarray, inner_params:dict | None=None) -> GridSearchCV:
        """
        The model implements the nested CV by entirely cloning a given GridSearchCV and performing a fit on it with a X and y data
        :param X_tr: the training data
        :param y_tr: the training labels
        :param inner_params: an utility dictionary for passing parameters during fit phase
        :return: GridSearchCV
        """
        gs = self.unfitted_model()
        gs.fit(X_tr, y_tr)

        inner_params = inner_params or {}
        silence_output = inner_params.get("silence_output", False)

        if not silence_output:
            grid_introspection(gs)

        return gs


    def predict(self, model: GridSearchCV, X: DataFrame) -> ndarray:
        """
        For a regression task we return the logits, or the output of the model
        :param model: the model
        :param X: the feature
        :return: the logits or the output of the model
        """
        return model.predict(X)


class SklearnRegressorRunner:
    """
    To be used in non-nested CV
    """

    def __init__(self, grid:GridSearchCV | BaseEstimator):
        """
        Utility class for wrapping a best estimator that needs to be instantiated and fitted
        from scratch and then being used in predictions.
        The grid is used to derive the estimator which is cloned first in 'init' and at each new_model invocation to avoid fitting or modification
        of the model made externally.
        Steps are:
            - runner = SklearnRegressorRunner(grid)
            - model = runner.new_model()
            - runner.fit(model, X, y)
            - runner.predict(model, X)
        :param grid: a fitted GridSearchCV object. Its best_estimator_ is cloned
                 and used as the fixed model template for non-nested CV.
        """
        # Model-selection notebooks pass an already fitted GridSearchCV, while
        # consumers such as an ensemble may already have the selected fixed
        # estimator.  In both cases keep an unfitted clone as template.
        estimator = grid.best_estimator_ if hasattr(grid, "best_estimator_") else grid
        self._base_estimator = clone(estimator)

    def unfitted_model(self) -> BaseEstimator:
        """
        Return a cloned unfitted estimator
        :return: unfitted estimator
        """
        return cast(BaseEstimator, clone(self._base_estimator))

    def fitted_model(self, X_tr: DataFrame, y_tr: ndarray, inner_params:dict=None) -> BaseEstimator:
        """
        Fit an estimator with a feature and a target. The runner invokes unfitted_model internally
        :param X_tr: the training features
        :param y_tr: the training targets
        :param inner_params: an eventual dictionary to be used for tuning the fitting phase
        :return: a fitted estimator
        """

        model = self.unfitted_model()
        model.fit(X_tr, y_tr)
        return model

    def predict(self, model:BaseEstimator, X: DataFrame) -> ndarray:
        """
        Predict using the fitted estimator
        :param model: the fitted estimator
        :param X: the features
        :return: the predictions
        """
        return model.predict(X)


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


def _loss_helper(value:any):

    return value


def kfold_losses(y_true: ndarray,y_pred:ndarray, loss_functions:dict) -> dict:
    """
    Compute kfold losses for a given dictionary
    :param y_true: true labels
    :param y_pred: predicted labels
    :param loss_functions: a dictionary of loss functions {'mee':( <function}, <reducer>)
    :return: a dictionary of losses per metric {'mee': 0.2239, 'mse': 0.5672, ...}
    """

    losses = dict()
    for loss_name, (loss_function,_) in loss_functions.items():
        batch_loss = _loss_helper(loss_function(y_true, y_pred))
        losses[loss_name] = batch_loss #+= batch_loss * batch_size

    return losses


def kfold(runner, X, y, folder_strategy, metrics: dict) -> FoldResults:
    """
    Run a kfold cross validation based on passed parameters.
    :param runner: the regressor runner
    :param X: the training features
    :param y: the training targets
    :param folder_strategy: the CV folder strategy
    :param metrics: the metrics to calculate
    :return: a FoldResults object
    """

    Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    yn = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    loss_dict = metrics
    metrics_key = list(set(metrics.keys()))

    # loop folds
    fold_results = FoldResults()
    for fold_nr, (tr_idx, vl_idx) in enumerate(folder_strategy.split(X), start=1):

        X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
        y_tr, y_vl = yn[tr_idx], yn[vl_idx]

        print(f"  ")
        print(f"Perform fold: {fold_nr}, train size: {len(y_tr)}, val size: {len(vl_idx)}")

        model = runner.fitted_model(X_tr, y_tr)

        y_pred_tr = runner.predict(model, X_tr)
        y_pred_vl = runner.predict(model, X_vl)

        all_losses = {
            'tr': kfold_losses(y_tr, y_pred_tr, loss_dict),
            'vl': kfold_losses(y_vl, y_pred_vl, loss_dict),
        }

        # Print fold summary
        print(f" ")
        print(f"Fold summary")
        print("-" * 80)
        for key, value in all_losses['vl'].items():
            print(f"{f'Validation {key.upper()}':<20} | {value:.4f}")
        print(f"{'Train':<20} | samples: {len(y_tr)}")
        print(f"{'Validation':<20} | samples: {len(y_vl)}")

        # Dinamically building a FoldResult
        fr = FoldResult()
        fr.set_metric(FOLD_NR, fold_nr)
        # Keep the information required to reconstruct strict out-of-fold
        # predictions. Existing consumers only reading metrics are unaffected.
        fr.set_metric("vl_indices", np.asarray(vl_idx))
        fr.set_metric("vl_predictions", np.asarray(y_pred_vl))
        for prefix in ("tr", "vl"):
            for metric in metrics_key:
                key = f"fold_{prefix}_{metric}"
                value = all_losses[prefix][metric]
                fr.set_metric(key, value)

        fold_results.append(fr)

    return fold_results


def mse_from_probability(logits:torch.Tensor, targets:torch.Tensor):
    """
    Computes the Mean Squared Error between the predicted probabilities
    and the binary target values.

    Since the neural network returns raw logits, the sigmoid function is
    applied first to convert them into probabilities in the [0, 1] range.
    The resulting MSE is used as an additional evaluation metric, while
    BCEWithLogitsLoss remains the objective function used for training.
    :param logits: the predicted probabilities, torch.Tensor, Raw outputs produced by the neural network.
    :param targets: the target values, torch.Tensor, Ground-truth binary labels.
    :return: the Mean Squared Error between predicted probabilities and targets.
    """
    probabilities = torch.sigmoid(logits)
    return F.mse_loss(probabilities, targets)


def mse(y_true, y_pred) -> float:
    """
    Implements mean squared error
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :return: the mean squared error
    """
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred) -> float:
    """
    Implements root mean squared error
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :return: the root mean squared error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    """
    Implements mean absolute error
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :return: the mean absolute error
    """
    return mean_absolute_error(y_true, y_pred)


def r2_mee(y_true, y_pred) -> float:
    """
    MEE-based relative score:
    1 = perfect
    0 = equal to mean baseline
    < 0 = worse than baseline
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    baseline_pred = np.tile(np.mean(y_true, axis=0), (len(y_true), 1))

    mee_model = mee(y_true, y_pred)
    mee_baseline = mee(y_true, baseline_pred)

    return 1.0 - (mee_model / mee_baseline)


def r2(y_true, y_pred) -> float:
    """
    Implements R squared
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :return: the R squared
    """
    return r2_score(y_true, y_pred)


def kfold_summary(fold_results: FoldResults, use: str= "mee") -> DataFrame:
    """
    Build a K-Fold summary table for a specific metric.
    :param fold_results: list[dict]
        Output histories from training (one per fold).
    :param use: metric to extract from fold results such as "mee", "rmse, etc..
    :return: pandas.DataFrame
        Table with per-fold metrics + mean/std for TR, VL and gap TR - VL
    """

    folds = []
    trs = []
    vls = []
    gaps = []
    gapsp = []
    for i, fold_result in enumerate(fold_results, start=1):

        tr_value = getattr(fold_result,f'fold_tr_{use}')
        vl_value = getattr(fold_result,f'fold_vl_{use}')
        # rel_gap = round((abs(gap) / tr_value) * 100,2)
        gap = vl_value - tr_value

        eps = 1e-12
        if abs(tr_value) < eps:
            rel_gap = None
        else:
            rel_gap = round((abs(gap) / abs(tr_value)) * 100, 2)

        folds.append(i)
        trs.append(tr_value)
        vls.append(vl_value)
        gaps.append(gap)
        gapsp.append(rel_gap)

    df = pd.DataFrame({
        "Fold": folds,
        "TR": trs,
        "VL": vls,
        "|TR-VL|": gaps,
    }).set_index("Fold")

    # Add mean and std rows
    df.loc["MEAN"] = df.mean()
    df.loc["STD"] = df.std()
    #df["Rel Gap(TR-VL)%"] = [f"{x:.2f}%" for x in df["Rel Gap(TR-VL)%"]]

    return df


def assess_sklearn_cv_robustness(cv, model, X, y, scoring="accuracy", seeds:list=[10, 20, 30, 40, 50]):
    """
    Show variability of sklearn scoring by running cross_val_score with varable seed
    :param cv: cross validation object
    :param model: the model
    :param X: training data
    :param y: training labels
    :param scoring: metric to be used
    :param seeds: list of seeds to try
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


def apply_score_correction(scoring, values:tuple, neg_score_list:tuple=(MEE, MAE, MSE, RMSE)) -> tuple:
    """
    Apply score correction in case of scoring starting with neg_ or if the scorer has a sign.
    :param scoring: the scorer identifier
    :param values: the list of float to correct
    :param neg_score_list: the list of scores to be recognized as negative
    :return: a list of floats
    """

    # Convert from negative score to positive loss for readability
    if isinstance(scoring, str):
        if scoring.startswith("neg_") or scoring in neg_score_list:
            return tuple([-1*x for x in values])
    # or... if it has sign we use it
    elif hasattr(scoring,"_sign"):
        return tuple([scoring._sign*x for x in values])

    # no action in other cases
    return values


def plot_learning_curve(model, X, y, cv, ax=None, scoring="accuracy", title:str="Learning curve", train_sizes=None):
    """
    Plot a learning curve for a given model.
    If scoring is a neg_* loss, the curve is plotted as the corresponding positive loss.
    """

    random_state = getattr(cv, "random_state", None)

    # if safe_train_sizes is None:
    #     train_sizes = safe_train_sizes
    # else:
    #     train_sizes = np.linspace(0.2, 1.0, 25),   # più punti

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        train_sizes= np.linspace(0.2, 1.0, 25) if train_sizes is None else train_sizes,   # più punti
        shuffle=True,
        random_state=random_state,
        n_jobs=-1
    )

    train_mean, val_mean = apply_score_correction(
        scoring,
        (train_scores.mean(axis=1), val_scores.mean(axis=1))
    )

    train_std, val_std = apply_score_correction(
        scoring,
        (train_scores.std(axis=1), val_scores.std(axis=1))
    )

    y_label = "Score"
    train_label = "Training score"
    val_label = "CV score"

    if isinstance(scoring, str) and scoring.startswith("neg_"):
        if scoring == "neg_mean_absolute_error":
            y_label = "MAE"
        elif scoring == "neg_mean_squared_error":
            y_label = "MSE"
        elif scoring == "neg_root_mean_squared_error":
            y_label = "RMSE"
        else:
            y_label = scoring.replace("neg_", "").replace("_", " ").upper()

        train_label = f"Training {y_label}"
        val_label = f"CV {y_label}"

    elif hasattr(scoring, "_sign"):
        y_label = "MEE"

    elif scoring == "accuracy":
        y_label = "Accuracy"
        train_label = "Training accuracy"
        val_label = "CV accuracy"

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(train_sizes_abs, train_mean, marker="o", markersize=0,label=train_label)
    ax.plot(train_sizes_abs, val_mean, marker="o", markersize=0, label=val_label)

    # bande di variabilità
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15
    )
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15
    )

    ax.set_xlabel("Training set size")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def generate_bootstrap_samples_from_dataset(X:pd.DataFrame, y:np.ndarray, n_samples:int=100, random_state:int=42)->[(pd.DataFrame,np.ndarray,float)]:
    """
    Generate bootstrap samples following bootstrap methodology
    :param X: the feature
    :param y: the target
    :param n_samples: how many samples we want
    :param random_state: the random state for shuffling
    :return: an array botstrap samples where each sample is a tuple (pd.DataFrame,np.ndarray,float)
    """

    rng = np.random.default_rng(random_state)

    n = len(X)
    samples = []

    for _ in range(n_samples):
        idx_boot = rng.integers(0, n, size=n)
        X_boot, y_boot = X.iloc[idx_boot], y[idx_boot]
        samples.append((X_boot, y_boot, idx_boot))

    return samples


def bootstrap_out_of_bag_scores(runner, X: pd.DataFrame, y:np.ndarray, samples: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]], metrics_fn, bootstrap_params:dict=None) -> list:
    """
    Analyze bootstrap variance
    :param model: the model
    :param X: features
    :param y: targets
    :param samples: the bootstrap samples
    :return: an array of Metrics
    """

    scores = []
    n = len(X)
    for sample in samples:

        X_boot, y_boot, idx_boot = sample

        # Return Out-Of-Bag (OOB) index, in other words the index not present in the samples
        selected = np.zeros(n, dtype=bool)
        selected[idx_boot] = True
        oob_idx = np.where(~selected)[0]

        # rare but it can happens
        if len(oob_idx) == 0: continue

        # dataset OOB
        X_oob = X.iloc[oob_idx]
        y_oob = y[oob_idx]

        scaler_template = bootstrap_params.get("scaler")
        scaler = clone(scaler_template) if scaler_template is not None else None
        if scaler is not None:
            X_boot_scaled = scaler.fit_transform(X_boot)
            X_oob_scaled = scaler.transform(X_oob)
        else:
            X_boot_scaled = X_boot
            X_oob_scaled = X_oob

        # fit model on bootstrap data
        fitted_model = runner.fitted_model(X_boot_scaled, y_boot, bootstrap_params)
        #idx = 2 if fitted_model.model_type=="classification" else 0
        y_pred = runner.predict(fitted_model, X_oob_scaled)

        # score
        score = dict()
        for key,(loss_function,_) in metrics_fn.items():
            score[key] = loss_function(y_oob, y_pred)
        scores.append(score)

    return scores


def aggregate_scores(scores:dict):

    header = set()
    values = dict()
    for m in scores:
        for key,value in m.items():
            header.add(key)
            if key not in values: values[key] = []
            values[key].append(value)

    means = []
    stds = []
    perc = []
    for key in header:
        v = values[key]
        mean = np.mean(v)
        std = np.std(v)
        means.append(mean)
        stds.append(std)
        perc.append((std/mean)*100)

    return pd.DataFrame({
        "Metric": [m.upper() for m in list(header)],
        "Mean": means,
        "Std": stds,
        "%": perc
    }).round(2)


def plot_bootstrap_distribution(results, title:str="Bootstrap Distribution"):

    sns.set_theme(style="whitegrid")

    plt.figure()

    ax = sns.histplot(results, kde=True, color="skyblue", stat="density", bins=15)

    mean_val = np.mean(results)
    std_val = np.std(results)
    # mean
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    # std
    plt.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.1, color='red', label=f'Std: {std_val:.2f}')

    plt.title(title)
    plt.xlabel("Mean")
    plt.ylabel("Density")
    plt.legend()

    plt.show()


class CustomRBFKernel:
    def __init__(self, gamma=0.1):
        self._gamma = gamma

    # def __call__(self, X, Y):
    #     X = np.asarray(X)
    #     Y = np.asarray(Y)
    #
    #     X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
    #     Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
    #     sq_dist = X_sq + Y_sq - 2 * X @ Y.T
    #
    #     return np.exp(-self.gamma * sq_dist)

    def __call__(self, X, Y, alpha=0.5, beta=1.0):
        X = np.asarray(X)
        Y = np.asarray(Y)

        linear_part = X @ Y.T
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)

        sq_dist = X_sq + Y_sq - 2 * X @ Y.T
        rbf_part = np.exp(-self._gamma * sq_dist)
        return alpha * linear_part + beta * rbf_part

    def get_params(self, deep=True):
        return {"gamma": self._gamma}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
