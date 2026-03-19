import copy
import datetime
import os

import seaborn as sns
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import ndarray
from sklearn import clone
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
import cross_common as cr
from cross_common import (MEE,MAE,MSE,RMSE)
from cross_common import (
    FOLD_NR,
    FOLD_TR_MSE,FOLD_VL_MSE,
    FOLD_TR_RMSE,FOLD_VL_RMSE,
    FOLD_TR_MEE,FOLD_VL_MEE,
    FOLD_TR_MAE,FOLD_VL_MAE,
    FOLD_TR_ACC,FOLD_VL_ACC)
from cross_common import (
    EPOCHS_TR_MSE, EPOCHS_VL_MSE,
    EPOCHS_VL_ACC, EPOCHS_TR_ACC,
    EPOCHS_TR_MAE, EPOCHS_VL_MAE,
    EPOCHS_TR_MEE, EPOCHS_VL_MEE,
    EPOCHS_TR_RMSE, EPOCHS_VL_RMSE,
    EPOCHS_TR_MSE_MEAN,EPOCHS_TR_MSE_STD,
    EPOCHS_VL_MSE_MEAN,EPOCHS_VL_MSE_STD,
    EPOCHS_TR_ACC_MEAN,EPOCHS_TR_ACC_STD,
    EPOCHS_VL_ACC_MEAN,EPOCHS_VL_ACC_STD,
    EPOCHS_TR_MAE_MEAN, EPOCHS_TR_MAE_STD,
    EPOCHS_VL_MAE_MEAN,EPOCHS_VL_MAE_STD,
    EPOCHS_TR_MEE_MEAN,EPOCHS_TR_MEE_STD,
    EPOCHS_VL_MEE_MEAN,EPOCHS_VL_MEE_STD)


# Default train params
DEFAULT_TRAIN_EPOCHS = 500
DEFAULT_TRAIN_PATIENCE = -1 # lower than 0 disable early stopping
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_TRAIN_DELTA = 1e-4
DEFAULT_TRAIN_VAL_RATIO = 0.2
DEFAULT_TRAIN_SEED = 1


class MLP(nn.Module):
    """
    Implements a skeleton of a Multilayer Perceptron network with given parameters
    """

    def __init__(self, net: nn.Module, adapter):
        """
        MLP constructor
        :param net: the MLP architecture in form of Sequence
        """
        super().__init__()
        self._net = net
        self._adapter = adapter

    def forward(self, x):
        """
        Implements the forward pass of the MLP architecture
        :param x: the input
        :return: a tensor containing the logits produced by the network, before any output activation function (e.g. sigmoid)
        """
        # The forward() method of Sequential accepts any input and forwards it to the first module it contains.
        # It then "chains" outputs to inputs sequentially for each subsequent module,
        # finally returning the output of the last module.
        # It is basically the forward pass of the whole architecture.
        return self._net(x)

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        self.eval()
        logits = self.forward(x)
        return self._adapter.probs(logits)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        logits = self.forward(x)
        return self._adapter.predict(logits)


class FeatureTargetSet:
    """
    This class is a wrapper for a dataset that is represented with X (the feature), and y (the labels).
    """

    def __init__(self, X: pd.DataFrame, y: ndarray):
        self._X = X.astype(np.float32).to_numpy()
        self._y = y
        return

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y


class TrainResults:
    """
    Helper class for model training results
    """

    def __init__(self, results: dict):
        """
#       Param results is a dictionary containing:
                - epochs_tr_loss: Training loss history. Sequence containing the value of the loss function computed on the training set at each training epoch. It is used to monitor how well the model fits the training data and to analyze the learning dynamics over time.
                - epochs_vl_loss: Validation loss history. Sequence containing the value of the loss function computed on the validation set at each training epoch. This quantity is used to assess the generalization capability of the model and to detect phenomena such as overfitting or underfitting.
                - epochs_tr_acc: Training accuracy history. Sequence containing the classification accuracy measured on the training set at each epoch. This metric represents the proportion of correctly classified samples and is meaningful only for classification tasks (Note: in regression tasks this quantity is not used).
                - epochs_vl_acc: Validation accuracy history. Sequence containing the classification accuracy measured on the validation set at each epoch. It is used to evaluate classification performance on unseen data and to monitor possible overfitting during training (Note: in regression tasks this quantity is not meaningful and is therefore ignored).
                - epochs_tr_mae: Training Mean Absolute Error (MAE) history. Sequence containing the MAE computed on the training set at each epoch. MAE measures the average absolute difference between predicted and target values and provides an error estimate less sensitive to outliers than MSE. It is meaningful for regression tasks (for classification it is typically not used).
                - epochs_vl_mae: Validation Mean Absolute Error (MAE) history. Sequence containing the MAE computed on the validation set at each epoch. It is used to monitor generalization in regression tasks and to detect overfitting/underfitting trends when compared with the training MAE curve.
                - epochs_tr_mee: Training Mean Euclidean Error (MEE) history. Sequence containing the MEE computed on the training set at each epoch. For multi-output regression, MEE is defined as the average Euclidean distance between the predicted output vector and the target vector for each sample; it summarizes the global prediction error across all output dimensions.
                - epochs_vl_mee: Validation Mean Euclidean Error (MEE) history. Sequence containing the MEE computed on the validation set at each epoch. This is the key metric for the CUP task (multi-output regression) and is used to select models/hyperparameters by tracking the best generalization performance across epochs.
                - epochs_grad: Gradient norm history. Sequence containing the norm of the gradient of the loss function with respect to the model parameters, computed during training. This diagnostic quantity is useful to analyze optimization stability and to detect issues such as vanishing or exploding gradients.
        :param results: the initialization dictionary
        """

        self._epochs_tr_mse, self._epochs_vl_mse = results.get(EPOCHS_TR_MSE), results.get(EPOCHS_VL_MSE)
        self._epochs_tr_acc, self._epochs_vl_acc = results.get(EPOCHS_TR_ACC), results.get(EPOCHS_VL_ACC)
        self._epochs_tr_mae, self._epochs_vl_mae = results.get(EPOCHS_TR_MAE), results.get(EPOCHS_VL_MAE)
        self._epochs_tr_mee, self._epochs_vl_mee = results.get(EPOCHS_TR_MEE), results.get(EPOCHS_VL_MEE)
        self._epochs_grad = results.get("epochs_grad")

        self._min_vl_mee = min(self._epochs_vl_mee)
        self._max_vl_mee = max(self._epochs_vl_mee)
        self._best_mee_epoch = self._epochs_vl_mee.index(self._min_vl_mee) + 1

    @property
    def min_vl_mee(self) -> float:
        """
        The min validation MEE
        :return:
        """
        return self._min_vl_mee

    @property
    def max_vl_mee(self) -> float:
        """
        The max validation MEE
        :return: float
        """
        return self._max_vl_mee

    @property
    def best_mee_epoch(self) -> int:
        """
        The epoch with best MEE metric
        :return: the epoch
        """
        return self._best_mee_epoch

    @property
    def epochs_tr_mse(self):
        """
        Training loss history. Sequence containing the value of the loss function computed on the training set at each training epoch. It is used to monitor how well the model fits the training data and to analyze the learning dynamics over time.
        :return:
        """
        return self._epochs_tr_mse

    @property
    def epochs_vl_mse(self):
        """
        Validation loss history. Sequence containing the value of the loss function computed on the validation set at each training epoch. This quantity is used to assess the generalization capability of the model and to detect phenomena such as overfitting or underfitting.
        :return:
        """
        return self._epochs_vl_mse

    @property
    def epochs_tr_acc(self):
        """
        Training accuracy history. Sequence containing the classification accuracy measured on the training set at each epoch. This metric represents the proportion of correctly classified samples and is meaningful only for classification tasks (Note: in regression tasks this quantity is not used).
        :return:
        """
        return self._epochs_tr_acc

    @property
    def epochs_vl_acc(self):
        """
        Validation accuracy history. Sequence containing the classification accuracy measured on the validation set at each epoch. It is used to evaluate classification performance on unseen data and to monitor possible overfitting during training (Note: in regression tasks this quantity is not meaningful and is therefore ignored).
        :return:
        """
        return self._epochs_vl_acc

    @property
    def epochs_tr_mae(self):
        """
        Training Mean Absolute Error (MAE) history. Sequence containing the MAE computed on the training set at each epoch. MAE measures the average absolute difference between predicted and target values and provides an error estimate less sensitive to outliers than MSE. It is meaningful for regression tasks (for classification it is typically not used).
        :return:
        """
        return self._epochs_tr_mae

    @property
    def epochs_vl_mae(self):
        """
        Validation Mean Absolute Error (MAE) history. Sequence containing the MAE computed on the validation set at each epoch. It is used to monitor generalization in regression tasks and to detect overfitting/underfitting trends when compared with the training MAE curve.
        :return:
        """
        return self._epochs_vl_mae

    @property
    def epochs_tr_mee(self):
        """
        Training Mean Euclidean Error (MEE) history. Sequence containing the MEE computed on the training set at each epoch. For multi-output regression, MEE is defined as the average Euclidean distance between the predicted output vector and the target vector for each sample; it summarizes the global prediction error across all output dimensions.
        :return:
        """
        return self._epochs_tr_mee

    @property
    def epochs_vl_mee(self):
        """
        Validation Mean Euclidean Error (MEE) history. Sequence containing the MEE computed on the validation set at each epoch. This is the key metric for the CUP task (multi-output regression) and is used to select models/hyperparameters by tracking the best generalization performance across epochs.
        :return:
        """
        return self._epochs_vl_mee

    @property
    def epochs_grad(self):
        """
        Gradient norm history. Sequence containing the norm of the gradient of the loss function with respect to the model parameters, computed during training. This diagnostic quantity is useful to analyze optimization stability and to detect issues such as vanishing or exploding gradients.
        :return:
        """
        return self._epochs_grad


def _make_split(dataset: FeatureTargetSet, val_ratio: float, seed:int) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Split a given set into training and validation sets according to val_ration and seed.
    Features and labels are often referred as X (uppercase) and y in NN literature,
    so we kept the same naming convention for function input variables.
    :param dataset: the TrainSet
    :param val_ratio: how many samples we want to reserve for validation. i.e. 0.2 = 20%
    :param seed: a fixed seed is mandatory for same results between runs, seed variation might lead to different learning curves
    :return: A tuple containing the training and validation sets (TR Features, TR Labels, VL Features, VL Labels)
    """

    X = dataset.X
    y = dataset.y

    rng = np.random.default_rng(seed) # <-- a fixed seed is mandatory
    idx = np.arange(len(X))
    rng.shuffle(idx)

    # Total number of samples that will enter into VL set
    n_val = int(len(X) * val_ratio)
    vl_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    all_data = X[tr_idx], y[tr_idx], X[vl_idx], y[vl_idx]
    return all_data


def _to_tensors(X: ndarray, y: ndarray) -> (ndarray, ndarray):
    """
    Convert X and y to tensor for use with PyTorch
    :param X: The features
    :param y: The labels
    :return: X and y as tensors (X: (N,D), y: (N,1) for scalar targets or (N,K) for multi-output targets)
    """

    # Forcing array to float
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Ensure y has shape (N, K)
    # - For scalar targets: (N,)  -> (N,1)
    # - For multi-output targets: (N,K) stays (N,K)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2:
        pass
    else:
        raise ValueError(f"y must be 1D or 2D array, got shape {y.shape}")

    # X and y torched
    return torch.from_numpy(X), torch.from_numpy(y)


def _make_loaders(X_tr: torch.Tensor, y_tr: torch.Tensor, X_vl: torch.Tensor, y_vl: torch.Tensor, batch_size: int):
    """
    Implements a batch stochastic gradient descent, handles data shuffling across epochs,
    and defines the notion of an epoch in the training procedure.
    The training set is shuffled in order to break any unwanted order or similiarity between samples.
	On the opposite the validation is instead kept unshuffled.
    :param X_tr:
    :param y_tr:
    :param X_vl:
    :param y_vl:
    :param batch_size:
    :return:
    """

    # Given a set of features and labels associated with them
    # the TensorDataset function aggregates in a single object
    # allowing to retrieve a sample with a single index
    # i.e. dataset[i]  →  (X[i], y[i])
    ds_tr = TensorDataset(X_tr, y_tr)
    ds_vl = TensorDataset(X_vl, y_vl)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True) # <- shuffle
    dl_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False) # <- no shuffle

    return dl_tr, dl_vl


def _compute_loss(model:MLP, X:torch.Tensor, y: torch.Tensor, loss_function: nn.Module) -> torch.Tensor:
    """
    This function computes the loss for a given set by invoking 'forward' on the model and applying the loss function.
    It is mandatory that the output layer of the model is kept linear.
    :param model: MLP model with linear output
    :param X: the features as tensor
    :param y: the labels as tensor
    :param loss_function: The given loss algorithm as specified in torch (i.e. BCEWithLogitsLoss(), MSELoss(), etc.)
    :return: a tensor containing the loss i.e. tensor(0.2408)
    """

    # The 'output' variable corresponds to the net value
    # of the last perceptron in the network.
    output = model.forward(X)

    # Our loss function, with y representing the target
    # to be compared with the output
    loss = loss_function(output, y)
    return loss


def epoch_accuracy(model, dataloader):
    """
    Calculate the epoch accuracy for a given model
    :param model:
    :param dataloader:
    :return:
    """

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            # logits = model(X)
            # preds = (torch.sigmoid(logits) > 0.5).float()
            # correct += (preds == y).sum().item()
            # total += y.size(0)
            preds = model.predict(X) # <-- reusing the adapter
            correct += (preds[0] == y).sum().item()
            total += y.size(0)

    if correct/total > 1:
        print("Warning, accuracy override")

    return correct / total


def epoch_loss(model: MLP, dataloader: DataLoader, loss_function: nn.Module) -> float:
    """
    This function computes the average loss over an entire dataset by aggregating the batch-wise losses
    without updating the model parameters, providing a stable estimate of training or validation error per epoch.
    :param model: the NN model
    :param dataloader: dataloader
    :param loss_function: the chosen algorithm (i.e. BCEWithLogitsLoss(), MSELoss(), etc.)
    :return: the average loss
    """

    # Store training mode to restore later
    was_training = model.training
    model.eval()

    model.train(False) # set model to evaluation only
    total_loss = 0.0
    total_n = 0

    # Disabling the gradient (i.e. no backprop)
    with torch.no_grad():
        # Cicling trough X our data and y the target labels
        for X, y in dataloader:

            number_of_labels = y.size(0)

            # Calculating the loss
            # In this case the loss is a 0-dimension Tensor obtained
            # from our loss function (see function documentation)
            # this is the reason why we use Tensor.item()
            loss = _compute_loss(model, X, y, loss_function).item()

            # The loss returned by PyTorch is the AVERAGE on the batch,
            # but we want to reconstruct the SUM of the losses on individual examples,
            # and then average the entire dataset, that's why we multiply for the nr of labels
            total_loss += loss * number_of_labels
            total_n += number_of_labels

    # Restore previous state
    if was_training:
        model.train()

    return total_loss / total_n


# Inference utility: run model.predict on a DataFrame
def predict(model: 'MLP', X_test: pd.DataFrame):
    """
    Run inference on a test set provided as a pandas DataFrame by reusing the model's adapter-based
    `MLP.predict(...)` method.
    The model is assumed to output logits in `forward()` and to expose `predict()` which returns hard labels
    (e.g., 0/1 for binary classification) according to the configured adapter.

    :param model: a trained MLP instance
    :param X_test: test features as a pandas DataFrame (already preprocessed / one-hot encoded)
    :param return_numpy: if True returns a 1D numpy array, otherwise returns a torch.Tensor
    :return: predictions for each sample (shape: (N,))
    """

    # Convert DataFrame to float32 numpy array and then to torch tensor
    X_np = X_test.astype(np.float32).to_numpy()
    X_t = torch.from_numpy(X_np)

    # Delegate the decision rule to the adapter via MLP.predict(...)
    preds_t = model.predict(X_t)  # shape (N,1) for binary adapters
    return preds_t[0]


class EarlyStoppingStrategy:
    """
    The class defines the early stopping strategy.
    """

    def __init__(self, patience:int = DEFAULT_TRAIN_PATIENCE, min_delta:float=DEFAULT_TRAIN_DELTA, metric:str=MEE):
        """
        Constructor
        :param patience: number of epochs to wait before stopping early
        :param min_delta: the delta between two consecutive epochs
        """
        self._patience = patience
        self._min_delta = min_delta
        self._metric = metric

    @property
    def patience(self):
        return self._patience

    @property
    def min_delta(self):
        return self._min_delta

    @property
    def metric(self):
        return self._metric

    def __str__(self):
        return f"EarlyStopping monitoring {self._metric} with patience={self._patience}, min_delta={self._min_delta}"


class SplitStrategy:

    def SplitStrategy(self):
        raise NotImplementedError

    def __init__(self):
        self._X_tr = None
        self._y_tr = None
        self._X_vl = None
        self._y_vl = None
        self._scaler = None

    @property
    def scaler(self):
        return self._scaler

    @property
    def train_set(self):
        return (self._X_tr, self._y_tr)

    @property
    def validation_set(self):
        return (self._X_vl, self._y_vl)


class HoldOutStrategy(SplitStrategy):
    """
    The classic split strategy to be applied to a single set, according to the ratio and seed
    """

    def __init__(self, train_set:FeatureTargetSet, val_ratio:float=DEFAULT_TRAIN_VAL_RATIO, seed:int=DEFAULT_TRAIN_SEED, scaler:TransformerMixin=None):
        super().__init__()
        X_tr, y_tr, X_vl, y_vl = _make_split(train_set, val_ratio=val_ratio, seed=seed)

        # applying scaling if needed
        if scaler is not None:
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_vl = scaler.transform(X_vl)

        # Convert each sets to tensor for use with PyTorch
        self._X_tr, self._y_tr = _to_tensors(X_tr, y_tr)  # TR sets
        self._X_vl, self._y_vl = _to_tensors(X_vl, y_vl)  # VL sets

        return


class ManualSplitStrategy(SplitStrategy):
    """
    The split strategy has already been applied to a single set, according to the ratio and seed
    """

    def __init__(self, train_set:FeatureTargetSet, validation_set:FeatureTargetSet, scaler:TransformerMixin=None):

        super().__init__()

        X_tr, y_tr = train_set.X, train_set.y
        X_vl, y_vl = validation_set.X, validation_set.y

        # applying scaling if needed
        if scaler is not None:
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_vl = scaler.transform(X_vl)

        self._X_tr, self._y_tr = _to_tensors(X_tr, y_tr)  # TR sets
        self._X_vl, self._y_vl = _to_tensors(X_vl, y_vl)  # VL sets


def train_wrapper_fn(
    model: MLP, datasets: SplitStrategy, optimizer_template, loss_function, batch_size: int | str,
    epochs:int, early_stopping_strategy:EarlyStoppingStrategy = None,
    scheduler_template= None,silence_output:bool=False
    ):

    return train(
        model,
        datasets,
        optimizer_template,
        loss_function,
        batch_size,
        epochs,
        early_stopping_strategy,
        scheduler_template,
        silence_output
    )


def predict_wrapper_fn(model, X):
    # forward + numpy, coerente col tuo adapter di regressione
    X = np.asarray(X, dtype=np.float32)
    with torch.no_grad():
        X_t = torch.from_numpy(X)
        _, scores = model.predict(X_t)
        return scores.cpu().numpy()


def train(model: MLP, split_strategy: SplitStrategy, optimizer_template, loss_function, batch_size: int | str,
          epochs:int, early_stopping_strategy:EarlyStoppingStrategy = None,
          scheduler_template= None, silence_output:bool=False) -> TrainResults:
    """
    This function trains the neural network for a fixed number of epochs using parametrized batch gradient descent,
    while monitoring performance on a validation set to track generalization and detect overfitting.
    :param scheduler_template: The learning decay strategy
    :param model: the model to be trained
    :param split_strategy: the split strategy (i.e. SplitByRatioStrategy, ManualSplitStrategy, etc.)
    :param optimizer_template:
    :param loss_function: the loss algorithm (i.e. BCEWithLogitsLoss(), MSELoss(), etc.)
    :param batch_size: "batch", "online" or the mini-batch size for training
    :param epochs: the number of epochs for training
    :param early_stopping_strategy: the early stopping strategy
    :param scheduler: learning rate scheduler
    :param silence_output: True/False if we want to display the train params
    :return: TrainResults
    """

    # Data container for later
    # visual representation
    epochs_tr_mse, epochs_vl_mse = [], []
    epochs_tr_acc, epochs_vl_acc = [], []
    epochs_tr_mae, epochs_vl_mae = [], []
    epochs_tr_mee, epochs_vl_mee = [], []
    epochs_grad_norm, epoch_grad_norms = [], []

    best_vl = float("inf")
    best_state = None # used only if early stopping is enabled
    bad_epochs = 0 # used only if early stopping is enabled
    metric_to_watch = early_stopping_strategy.metric

    # initialize optimizer
    optimizer = optimizer_template(model.parameters())
    if scheduler_template is not None:
        scheduler = scheduler_template(optimizer)
    else:
        scheduler = None

    X_tr, y_tr = split_strategy.train_set
    X_vl, y_vl = split_strategy.validation_set

    # Build DataLoader (or the batch as it is known in ML)
    if batch_size=="batch":
        batch_size = X_tr.shape[0] # if 'batch' then the batch size is the X dimension
    elif batch_size=="online":
        batch_size = 1

    dl_tr, dl_vl = _make_loaders(X_tr, y_tr, X_vl, y_vl, batch_size=batch_size)

    # Performing epochs loop
    for epoch in range(epochs):

        # Set model in training mode
        model.train()

        # Loop trough dataset
        for X, y in dl_tr:

            # Important!!! As PyTorch performs
            # a sum of all gradients by default
            # the following disables such capability
            optimizer.zero_grad()

            # Calculating the loss and applying the backpropagation
            loss = _compute_loss(model, X, y, loss_function)
            loss.backward()

            # Take gradient measurement before update
            epoch_grad_norms.append(gradient_norm(model))
            optimizer.step()

        # Loss estimation at the end of epoch
        epoch_tr_mse = epoch_loss(model, dl_tr, MSELoss(reduction="mean"))
        epochs_tr_mse.append(epoch_tr_mse)
        epoch_vl_mse = epoch_loss(model, dl_vl, MSELoss(reduction="mean"))
        epochs_vl_mse.append(epoch_vl_mse)

        # Calculate Mean Absolute Error (MAE)
        epoch_tr_mae = epoch_loss(model, dl_tr, nn.L1Loss(reduction="mean"))
        epochs_tr_mae.append(epoch_tr_mae)
        epoch_vl_mae = epoch_loss(model, dl_vl, nn.L1Loss(reduction="mean"))
        epochs_vl_mae.append(epoch_vl_mae)

        epoch_tr_mee = epoch_loss(model, dl_tr, MEELoss(reduction="mean"))
        epochs_tr_mee.append(epoch_tr_mee)
        epoch_vl_mee = epoch_loss(model, dl_vl, MEELoss(reduction="mean"))
        epochs_vl_mee.append(epoch_vl_mee)

        epochs_tr_acc.append(epoch_accuracy(model, dl_tr))
        epochs_vl_acc.append(epoch_accuracy(model, dl_vl))

        # Gradient mean
        epochs_grad_norm.append(float(np.mean(epoch_grad_norms)) if len(epoch_grad_norms) else 0.0)

        # Applying learning rate decay
        if scheduler is not None:
            scheduler.step(epoch_vl_mse)

        # ---- EARLY STOPPING LOGIC ----
        if early_stopping_strategy is not None:

            if metric_to_watch == MSE:
                epoch_vl = epoch_vl_mse
            elif metric_to_watch == MEE:
                epoch_vl = epoch_vl_mee
            elif metric_to_watch == MAE:
                epoch_vl = epoch_vl_mae
            else:
                raise ValueError(f"{metric_to_watch} is not supported")

            if epoch_vl < best_vl - early_stopping_strategy.min_delta:
                best_vl = epoch_vl
                bad_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1

            if bad_epochs >= early_stopping_strategy.patience:
                if not silence_output:
                    print(f"Early stopping at epoch {epoch} (best VL {metric_to_watch} loss: {best_vl:.4f})")
                break

        # Keeping track of the latest best model
        if best_state is not None:
            model.load_state_dict(best_state)
        # -----------------------------

    if not silence_output:
        _print_train_summary(model, dl_tr,dl_vl,optimizer_template,loss_function, scheduler_template, batch_size, epochs,early_stopping_strategy)

    return TrainResults({
        EPOCHS_TR_MSE: epochs_tr_mse, EPOCHS_VL_MSE: epochs_vl_mse,
        EPOCHS_TR_ACC: epochs_tr_acc, EPOCHS_VL_ACC: epochs_vl_acc,
        EPOCHS_TR_MAE: epochs_tr_mae, EPOCHS_VL_MAE: epochs_vl_mae,
        EPOCHS_TR_MEE: epochs_tr_mee, EPOCHS_VL_MEE: epochs_vl_mee,
        "epochs_grad": epochs_grad_norm
    })


def kfold(untrained_base_model: MLP, X, y, fold_strategy, inner_train_params: dict, scaler_template=None) -> cr.FoldResults:
    """
    Runs K-Fold cross-validation by reusing existing training/evaluation code.
    Returns per-fold histories and final validation metrics.
    Each fold is trained using the train() function.
    :param untrained_base_model: the untrained MLP model
    :param X: training data
    :param y: training labels
    :param fold_strategy: the fold strategy (i.e. KFold, StratifiedKfold,...)
    :param inner_train_params: the parameters for the inner trainer
    :param scaler_template: The scaler if needed. Note: the scaler will eventually be cloned in each fold
    """

    fold_results = cr.FoldResults()

    # Fold iteration
    for fold_nr, (tr_idx, vl_idx) in enumerate(fold_strategy.split(X, y)):

        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_vl, y_vl = X.iloc[vl_idx], y[vl_idx]

        silence_output = inner_train_params.get("silence_output", False)

        if not silence_output:
            print(f"  ")
            print(f"Perform fold: {fold_nr}, train size: {len(y_tr)}, val size: {len(vl_idx)}")

        # Model cloning
        fold_model = copy.deepcopy(untrained_base_model)

        # Training
        train_result = train(
            fold_model,
            ManualSplitStrategy(FeatureTargetSet(X_tr, y_tr), FeatureTargetSet(X_vl, y_vl), scaler=clone(scaler_template)),
            **inner_train_params
        )

        epochs_tr_mse, epochs_vl_mse = train_result.epochs_tr_mse, train_result.epochs_vl_mse
        epochs_tr_acc, epochs_vl_acc= train_result.epochs_tr_acc, train_result.epochs_vl_acc
        epochs_tr_mae, epochs_vl_mae= train_result.epochs_tr_mae, train_result.epochs_vl_mae
        epochs_tr_mee, epochs_vl_mee= train_result.epochs_tr_mee, train_result.epochs_vl_mee
        epochs_grad = train_result.epochs_grad

        # Data gathering
        fold_results.append(cr.FoldResult({
            FOLD_NR: fold_nr,

            # NN Specific attributes
            EPOCHS_TR_MSE: epochs_tr_mse, EPOCHS_VL_MSE: epochs_vl_mse,
            EPOCHS_TR_ACC: epochs_tr_acc, EPOCHS_VL_ACC: epochs_vl_acc,
            EPOCHS_TR_MAE: epochs_tr_mae, EPOCHS_VL_MAE: epochs_vl_mae,
            EPOCHS_TR_MEE: epochs_tr_mee, EPOCHS_VL_MEE: epochs_vl_mee,
            EPOCHS_TR_RMSE: epochs_tr_mse,EPOCHS_VL_RMSE: epochs_vl_mse,

            EPOCHS_TR_MSE_MEAN: np.mean(epochs_tr_mse), EPOCHS_TR_MSE_STD: np.std(epochs_tr_mse),
            EPOCHS_VL_MSE_MEAN: np.mean(epochs_vl_mse), EPOCHS_VL_MSE_STD: np.std(epochs_vl_mse),
            EPOCHS_TR_ACC_MEAN: np.mean(epochs_tr_acc), EPOCHS_TR_ACC_STD: np.std(epochs_tr_acc),
            EPOCHS_VL_ACC_MEAN: np.mean(epochs_vl_acc), EPOCHS_VL_ACC_STD: np.std(epochs_vl_acc),
            EPOCHS_TR_MAE_MEAN: np.mean(epochs_tr_mae), EPOCHS_TR_MAE_STD: np.std(epochs_tr_mae),
            EPOCHS_VL_MAE_MEAN: np.mean(epochs_vl_mae), EPOCHS_VL_MAE_STD: np.std(epochs_vl_mae),
            EPOCHS_TR_MEE_MEAN: np.mean(epochs_tr_mee), EPOCHS_TR_MEE_STD: np.std(epochs_tr_mee),
            EPOCHS_VL_MEE_MEAN: np.mean(epochs_vl_mee), EPOCHS_VL_MEE_STD: np.std(epochs_vl_mee),

            FOLD_TR_MSE: float(min(epochs_tr_mse)), FOLD_VL_MSE: float(min(epochs_vl_mse)),
            FOLD_TR_RMSE: float(min(epochs_tr_mse)), FOLD_VL_RMSE: float(min(epochs_vl_mse)),
            FOLD_TR_MEE: float(min(epochs_tr_mee)), FOLD_VL_MEE: float(min(epochs_vl_mee)),
            FOLD_TR_MAE: float(min(epochs_tr_mae)), FOLD_VL_MAE: float(min(epochs_vl_mae)),
            FOLD_TR_ACC: float(max(epochs_tr_acc)), FOLD_VL_ACC: float(max(epochs_vl_acc))
        }))

    return fold_results


def init_weights(m: nn.Module, method: str, nonlinearity: str = "relu") -> None:
    """
    Initializes the weights of a model layer according to the specified method.
    This function is intended to be passed to model.apply(...) and used only for learning purposes.
    In particular to verify the negative effect of setting the weight to zero.
    :param m: a module of the model (visited recursively by model.apply)
    :param method: initialization algorithm, only "kaiming" and "constant" are supported
    :param nonlinearity: activation function used after the layer (for Kaiming)
    """

    if not isinstance(m, nn.Linear):
        return

    if method == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
    elif method == "xavier":
        nn.init.xavier_uniform_(m.weight)

    # Not to be used, only for learning purposes
    elif method == "constant":
        nn.init.constant_(m.weight, 0.0)

    else:
        raise ValueError(f"Unknown initialization method: {method}")

    if m.bias is not None:
        nn.init.zeros_(m.bias)


def gradient_norm(model: torch.nn.Module) -> float:
    """
    This function computes the gradient norm of the model parameters.
    :param model: the NN model
    :return: gradient norm (float)
    """

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def max_grad(model: torch.nn.Module) -> float:
    """
    Computes the maximum absolute gradient across all model parameters.
    :param model: the NN model
    :return: max absolute gradient value (float)
    """

    max_g = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_max = p.grad.detach().abs().max().item()
            max_g = max(max_g, param_max)

    return max_g


class OutputAdapter:
    """
    Maps raw model outputs to probabilities/scores and then to predicted labels.
    """
    def __init__(self, link_fn, decision_fn):
        self._link_fn = link_fn          # logits -> probs/scores
        self._decision_fn = decision_fn  # probs/scores -> labels

    def predict(self, logits: Tensor) -> (Tensor, Tensor):
        """
        Predicts the class of the output given the logits
        :param logits: the input X Tensor
        :return: a tuple (predictions, scores) containing the predicted labels and probabilities
        """
        scores = self._link_fn(logits)
        predictions = self._decision_fn(scores)
        return predictions, scores


def binary_bce_adapter(threshold: float = 0.5) -> OutputAdapter:
    link = torch.sigmoid
    decision = lambda p: (p >= threshold).float()
    return OutputAdapter(link, decision)


def multiclass_ce_adapter(dim: int = 1) -> OutputAdapter:
    link = lambda z: torch.softmax(z, dim=dim)
    decision = lambda p: torch.argmax(p, dim=dim)
    return OutputAdapter(link, decision)


def regression_adapter() -> OutputAdapter:
    link = lambda z: z                 # identity
    decision = lambda z: z             # per regressione "pred" = valore
    return OutputAdapter(link, decision)


def plot_epoch_mee(epochs_tr_mee, epochs_vl_mee):

    plot_epochs_curves(
        [
            {"key": "TR MEE", "value": epochs_tr_mee},
            {"key": "VL MEE", "value": epochs_vl_mee}
        ],
        x_label="Epochs", y_label="MEE", title="Training vs Validation Loss"
    )


def plot_epoch_mse(epochs_tr, epochs_vl):

    plot_epochs_curves(
        [
            {"key": "TR MSE", "value": epochs_tr},
            {"key": "VL MSE", "value": epochs_vl}
        ],
        x_label="Epochs", y_label="Loss", title="Training vs Validation Loss"
    )


def plot_epoch_accuracy(epochs_tr_acc, epochs_vl_acc):

    plot_epochs_curves(
        [
            {"key": "TR Accuracy", "value": epochs_tr_acc},
            {"key": "VL Accuracy", "value": epochs_vl_acc}
        ],
        x_label="Epochs", y_label="Loss", title="Training vs Validation Accuracy"
    )


def plot_gradient_norm_line(grad_norms, step=10):

    epochs = np.arange(0, len(grad_norms), step)
    values = [grad_norms[i] for i in epochs]

    df = pd.DataFrame({
        "Epoch": epochs,
        "Gradient norm": values
    })

    plt.figure()
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Gradient norm",
        marker="o",
        markersize=4,
    )

    # --- MIN / MAX lines ---

    gmin = min(values)
    gmax = max(values)
    plt.axhline(
        gmin, linestyle="--", linewidth=1,
        label=f"Min = {gmin:.4f}"
    )
    plt.axhline(
        gmax, linestyle="--", linewidth=1,
        label=f"Max = {gmax:.4f}"
    )

    plt.title("Gradient norm (sampled every {} epochs)".format(step))
    plt.ylabel("Gradient")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_gradient_norm_bars(grad_norms, step=10):

    sns.set_theme(style="whitegrid")

    epochs = np.arange(0, len(grad_norms), step)
    values = [grad_norms[i] for i in epochs]

    df = pd.DataFrame({
        "Epoch": epochs,
        "Gradient norm": values
    })

    plt.figure()
    sns.barplot(
        data=df,
        x="Epoch",
        y="Gradient norm",
        color="steelblue"
    )

    plt.xlabel("Epoch")
    plt.ylabel(r"$\|\nabla L\|$")
    plt.title(f"Gradient norm (one bar every {step} epochs)")
    plt.tight_layout()
    plt.show()


def _print_train_summary(model, train_dataloader,validation_dataloader,optimizer_template,loss_function,scheduler_template, batch_size, epochs,early_stopping_strategy):
    """
    Display a nice and formatted table containing the train hyperparameters
    :param model: the model
    :param train_dataloader: the training dataloader
    :param validation_dataloader: the validation dataloader
    :param optimizer_template: the optimizer template
    :param loss_function: the loss function (MSELoss, etc.)
    :param scheduler_template:
    :param batch_size: the mini-batch or batch size
    :param epochs: the number of epochs
    :param early_stopping_strategy: the early stopping strategy
    :return: None
    """

    if batch_size == 1:
        batch_size_desc = "online"
    elif batch_size == len(train_dataloader.dataset):
        batch_size_desc = "batch"
    else:
        batch_size_desc = f"mini-batch ({batch_size})"

    # header
    print(f"Train summary")
    print("-" * 100)
    print(f"Epochs          | {epochs}")
    print(f"Train           | samples: {len(train_dataloader.dataset)}")
    print(f"Validation      | samples: {len(validation_dataloader.dataset)}")
    print(f"Batch type      | {batch_size_desc}")
    print(f"Loss function   | {loss_function}")
    print(f"Optimizer       | class: {optimizer_template.func} , params: {optimizer_template.keywords}")
    if scheduler_template is not None:
        print(f"LR Decay        | params: {scheduler_template.keywords}")
    else:
        print(f"LR Decay        | disabled")
    if early_stopping_strategy is not None:
        print(f"Early Stopping  | patience: {early_stopping_strategy.patience} , minimum delta: {early_stopping_strategy.min_delta}")
    else:
        print(f"Early Stopping  | disabled")
    print("-" * 100)


def plot_epochs_curves(plots: list[dict], x_label: str, y_label: str, title:str=None):

    rows = []

    for p in plots:
        for i, v in enumerate(p["value"]):
            rows.append({
                x_label: i,
                y_label: v,
                "Series": p["key"]
            })

    df = pd.DataFrame(rows)

    plt.figure()
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x=x_label, y=y_label, hue="Series")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


class MEELoss(nn.Module):
    """
    Custom implementation of Mean Euclidean Error (MEE) loss.
    :param reduction: Specifies the reduction to apply to the output
        - "mean": mean Euclidean error over the batch
        - "sum":  sum of Euclidean errors over the batch
        - "none": per-sample Euclidean error (shape: [batch_size])
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, y_hat, y):
        # Euclidean distance per sample
        distances = torch.linalg.vector_norm(y_hat - y, ord=2, dim=1)

        if self.reduction == "mean":
            return distances.mean()
        elif self.reduction == "sum":
            return distances.sum()
        else:
            return distances

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{type(self).__name__}(reduction='{self.reduction}') - custom function"


def build_mlp(input_dim: int, output_dim: int, hidden_units: list[int] , activation: str) -> nn.Sequential:
    """
    Build an MLP as nn.Sequential with configurable hidden layers.
    Example: hidden_units=[128,64] gives input->128->64->output.
    """
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU
    }
    if activation not in act_map:
        raise ValueError(f"Unknown activation: {activation}. Choose from {list(act_map.keys())}")

    Act = act_map[activation]

    layers = []
    prev = input_dim

    for h in hidden_units:
        layers.append(nn.Linear(prev, h, bias=True))
        layers.append(Act())
        prev = h

    # output layer (linear)
    layers.append(nn.Linear(prev, output_dim, bias=True))

    return nn.Sequential(*layers)


class TorchRegressorRunner:

    def __init__(self, model_template):
        self._model_template = model_template

    def new_model(self):
        return copy.deepcopy(self._model_template)

    def fit(self, model, X_tr, y_tr, bootstrap_params):

        #X_tr, y_tr = split_strategy.X_tr, split_strategy.y_tr
        #X_vl, y_vl = split_strategy.X_vl, split_strategy.y_vl
        train_params = bootstrap_params["train_params"]
        split_strategy_cls = bootstrap_params['split_strategy']
        split_strategy = split_strategy_cls(FeatureTargetSet(X_tr, y_tr))
        train(
            model,
            split_strategy,
            **train_params
        )

        return model

    def predict(self, model, X):
        return predict(model, X)


def suggest_trial_param(trial, name, space):
    """
    Suggest a trial parameter with a space
        - suggest_trial_param(trial, "learning_rate", [1e-2, 1e-3, 1e-4, 1e-5]) --> suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])
        - suggest_trial_param(trial, "learning_rate", (0.0033, 0.003) --> suggest_float("learning_rate", 0.0033, 0.003, log=True) \n
         results in the interval [0.00033, 0.00045,0.00060,0.00080,0.00105,0.00140,0.00190,0.00250,0.003]
    :param trial: the trial
    :param name: the name of the attribute
    :param space: the space of the attribute
    :return: a suggestion
    """

    # categorical
    if isinstance(space, list):
        return trial.suggest_categorical(name, space)

    # float range
    if isinstance(space, tuple) and len(space) == 2:
        low, high = space
        return trial.suggest_float(name, low, high, log=True)

    raise ValueError(f"Invalid search space for {name}")


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


# def summarize_lr_from_folds(fold_results, epoch_metric:str=EPOCHS_VL_MEE, fold_metric:str=FOLD_VL_MEE, early_epoch:int=30):
#     """
#     fold_results: lista di dizionari, uno per fold
#     ognuno deve contenere almeno:
#         - EPOCHS_VL_MEE
#         - FOLD_VL_MEE
#     """
#     analyses = [
#         analyze_lr_curve(getattr(fold, epoch_metric), early_epoch=early_epoch)
#         for fold in fold_results
#     ]
#
#     return {
#         "mean_early_gain": float(np.mean([a["early_gain"] for a in analyses])),
#         "std_early_gain": float(np.std([a["early_gain"] for a in analyses])),
#
#         "mean_oscillation": float(np.mean([a["oscillation"] for a in analyses])),
#         "std_oscillation": float(np.std([a["oscillation"] for a in analyses])),
#
#         "mean_best": float(np.mean([getattr(fold, fold_metric) for fold in fold_results])),
#         "std_best": float(np.std([getattr(fold, fold_metric) for fold in fold_results])),
#
#         "mean_best_epoch": float(np.mean([a["best_epoch"] for a in analyses]))
#     }


def summarize_weight_decay_from_folds(fold_histories, vl_key:str=FOLD_VL_MEE, tr_key=FOLD_TR_MEE):

    mean_best_tr_metric = float(np.mean([getattr(fold, tr_key) for fold in fold_histories]))
    mean_best_vl_metric = float(np.mean([getattr(fold, vl_key) for fold in fold_histories]))
    gap_tr_vl_mee = mean_best_vl_metric - mean_best_tr_metric

    return {
        "mean_best_tr": mean_best_tr_metric,
        "mean_best_vl": mean_best_vl_metric,
        "gap_tr_vl": gap_tr_vl_mee
    }


def find_best_lr_from_trials(study) -> float:
    """
    The learning rate is selected according to three complementary criteria derived from the training dynamics observed during K-Fold cross-validation:
	    - Fast initial convergence: the training curve should decrease rapidly during the first 30–50 epochs, indicating that the optimizer is able to make effective progress at the beginning of training.
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
            -t.values[2]    # 3. maximize early_gain as final step when two or more trials share the same (1) and (2)
        )
    )

    return results.params["learning_rate"]


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
    return summary["mean_best"],summary["mean_oscillation"],summary["mean_early_gain"]


def find_best_weight_decay_from_trials(study) -> float:
    """
    Select the best weight decay from trials.
    Priority: (1) lowest mean validation metric, (2) smallest TR–VL gap.
    This favors configurations with strong validation performance and good generalization.
    :param study: the optuna study object
    :return: best weight_decay
    """
    the_trial = min(
        study.best_trials,
        key=lambda t: (
            t.values[0],   # mean_best_vl_metric
            t.values[1]    # gap_tr_vl
        )
    )
    return the_trial.params["weight_decay"]


def evaluate_wd(fold_histories: cr.FoldResults,vl_key:str,tr_key:str) -> tuple[float,float]:
    """
    Find evaluation values for optuna regarding weight decay.
    :param fold_histories:
    :param vl_key: for example fold_vl_mee
    :param tr_key: for example fold_tr_mee
    :return: best vl metric and best gap
    """
    mean_best_tr_metric = float(np.mean([getattr(fold, tr_key) for fold in fold_histories]))
    mean_best_vl_metric = float(np.mean([getattr(fold, vl_key) for fold in fold_histories]))
    gap_tr_vl = mean_best_vl_metric - mean_best_tr_metric
    return mean_best_vl_metric,gap_tr_vl


def save_run_report(model_baseline: str, net, inner_train_params: dict, metrics: dict, folder: str = "runs"):
    """
    Save an NN run
    :param model_baseline:
    :param net:
    :param inner_train_params:
    :param metrics:
    :param folder:
    :return:
    """

    # Create folder if not exists
    os.makedirs(folder, exist_ok=True)

    # Timestamp name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"run_{timestamp}.txt"
    path = os.path.join(folder, filename)

    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"RUN ID: run_{timestamp}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model:\n")
        f.write(f"{net}\n")
        f.write("\n")

        f.write("Hyperparameters:\n")
        for k, v in inner_train_params.items():
            f.write(f"  - {k}: {v}\n")

        f.write("\nMetrics (K-Fold mean ± std):\n")
        for k, (vl_mean, tr_mean) in metrics.items():
            f.write(f"  VL {k.upper()}: {vl_mean[0]:.4f} ± {vl_mean[1]:.4f}\n")
            f.write(f"  TR {k.upper()}: {tr_mean[0]:.4f} ± {tr_mean[1]:.4f}\n")
            f.write(f"  VL - TR: {(vl_mean[0] - tr_mean[0]):.4f}")
            f.write("\n")
            f.write("-"*30)
            f.write("\n")

        f.write(f"  Baseline: {model_baseline:.4f}")
        f.write("\n")
        f.write("-"*30)

        f.write("\n")

    return path


def find_best_architecture_from_trials(study) -> tuple[list[int],float,float]:
    """
    The best architecture is the one that minimize the vl metric and the gap between tr and vl metric
    :param study: the study of the trials
    :return: the best architecture
    """
    the_trial = min(
        study.best_trials,
        key=lambda t: (
            t.values[0],   # mean_best_vl_metric
            t.values[1]    # gap_tr_vl
        )
    )
    layers = [int(item) for item in the_trial.params["hidden_layers"].split(",")]
    return layers, the_trial.values[0], the_trial.values[1]


def evaluate_architecture(fold_histories: cr.FoldResults, vl_key:str, tr_key:str) -> tuple[float, float]:
    """
    Evaluate a neural network architecture using K-Fold training histories.
    The function computes two metrics aggregated across folds:
     - mean_best_vl: The mean validation metric (e.g., VL_MEE) obtained from the best epoch of each fold. This is the primary indicator of generalization performance.
     - overfit_gap: The difference between validation and training performance averaged across folds (VL − TR). Only positive values are considered, since a large positive gap indicates overfitting. Negative values are clipped to zero to avoid rewarding configurations where validation happens to be better than training.
    These two values are used during architecture search to favor models that:
     - achieve low validation error
     - do not exhibit significant overfitting.
    :fold_histories: fold results
    :vl_key: vl metric
    :tr_key: tr metric
    :return: the best vl metric and the overfit_gap
    """

    # Reusing common function
    mean_best_tr, std_best_tr = cr.extract_mean_std(fold_histories,tr_key)
    mean_best_vl, std_best_vl = cr.extract_mean_std(fold_histories,vl_key)

    # See function comment for the reason why
    overfit_gap = max((mean_best_vl - mean_best_tr), 0.0)

    return mean_best_vl,overfit_gap