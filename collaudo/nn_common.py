import copy
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import nn, Tensor
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader


# Default train params
DEFAULT_TRAIN_EPOCHS = 500
DEFAULT_TRAIN_PATIENCE = -1 # lower than 0 disable early stopping
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_TRAIN_DELTA = 1e-4
DEFAULT_TRAIN_VAL_RATIO = 0.2
DEFAULT_TRAIN_SEED = 1
DEFAULT_TRAIN_BATCH_SIZE = 32


def _make_split(X:ndarray, y:ndarray, val_ratio, seed) -> (ndarray, ndarray, ndarray, ndarray):
    """
    Split a given set into training and validation sets to perform a hold-out validation.
    Features and labels are often referred as X (uppercase) and y in NN literature,
    so we kept the same naming convention for function input variables.
    :param X: the features
    :param y: the labels
    :param val_ratio: how many samples we want to reserve for validation. i.e. 0.2 = 20%
    :param seed: a fixed seed is mandatory for same results between runs, seed variation might lead to different learning curves
    :return: A tuple containing the training and validation sets (TR Features, TR Labels, VL Features, VL Labels)
    """

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
    :return: X and y as tensors
    """

    # Forcing array to float
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)  # (N,1)

    # X and y torched
    return torch.from_numpy(X), torch.from_numpy(y)


def _prepare_for_torch(X:ndarray, y:ndarray, val_ratio:float, seed: int) ->(Tensor, Tensor, Tensor, Tensor):
    """
    This function prepare a dataset to be used with PyTorch by performing the following actions:
    1. Split into training and validation sets according to seed and ratio
    2. Transform each set into tensors
    :param X: the features
    :param y: the labels
    :param val_ratio: how many samples we want to reserve for validation. i.e. 0.2 = 20%
    :param seed: a fixed seed is mandatory for same results between runs, seed variation might lead to different learning curves
    :return: A tuple of tensors (X TR, y TR, X VL, y VL)
    """

    # Split data into training set and validation set
    X_tr, y_tr, X_vl, y_vl = _make_split(X, y, val_ratio=val_ratio, seed=seed)

    # Converto each sets to tensor for use with PyTorch
    X_tr_t, y_tr_t = _to_tensors(X_tr, y_tr) # TR sets
    X_vl_t, y_vl_t = _to_tensors(X_vl, y_vl) # VL sets

    return X_tr_t, y_tr_t, X_vl_t, y_vl_t


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


def compute_loss(model:MLP, X:torch.Tensor, y: torch.Tensor, loss_function: nn.Module) -> torch.Tensor:
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

    # Our loss function, here y represents the target
    # which is being compared with the output
    loss = loss_function(output, y)
    return loss


def epoch_accuracy(model, dataloader):
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

    return correct / total


def epoch_loss(model: MLP, dataloader: DataLoader, loss_algorithm: nn.Module) -> float:
    """
    This function computes the average loss over an entire dataset by aggregating the batch-wise losses
    without updating the model parameters, providing a stable estimate of training or validation error per epoch.
    :param model: the NN model
    :param dataloader: dataloader
    :param loss_algorithm: the chosen algorithm (i.e. BCEWithLogitsLoss(), MSELoss(), etc.)
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
            loss = compute_loss(model, X, y, loss_algorithm).item()

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
    return preds_t


def train(model: MLP, X: pd.DataFrame, y: ndarray, optimizer_template, loss_algorithm, batch_size: int | str = DEFAULT_TRAIN_BATCH_SIZE,
          epochs:int=DEFAULT_TRAIN_EPOCHS, val_ratio:float=DEFAULT_TRAIN_VAL_RATIO, seed:int=DEFAULT_TRAIN_SEED,
          patience:int=DEFAULT_TRAIN_PATIENCE, min_delta:float=DEFAULT_TRAIN_DELTA,
          scheduler_template= None,silence_output:bool=False) -> {}:
    """
    This function trains the neural network for a fixed number of epochs using parametrized batch gradient descent,
    while monitoring performance on a validation set to track generalization and detect overfitting.
    It returns a dictionary containing:
            - record_tr_loss Training loss history. Sequence containing the value of the loss function computed on the training set at each training epoch. It is used to monitor how well the model fits the training data and to analyze the learning dynamics over time.\n
            - record_vl_loss: Validation loss history. Sequence containing the value of the loss function computed on the validation set at each training epoch. This quantity is used to assess the generalization capability of the model and to detect phenomena such as overfitting or underfitting.
            - record_tr_acc: Training accuracy history. Sequence containing the classification accuracy measured on the training set at each epoch. This metric represents the proportion of correctly classified samples and is meaningful only for classification tasks. (Note: in regression tasks this quantity is not used.)
            - record_vl_acc: Validation accuracy history. Sequence containing the classification accuracy measured on the validation set at each epoch. It is used to evaluate classification performance on unseen data and to monitor possible overfitting during training (Note: in regression tasks this quantity is not meaningful and is therefore ignored).
            - record_grad_norm: Gradient norm history. Sequence containing the norm of the gradient of the loss function with respect to the model parameters, computed during training. This diagnostic quantity is useful to analyze optimization stability and to detect issues such as vanishing or exploding gradients.
    :param model: the model to be trained
    :param X: training features as a pandas DataFrame
    :param y: training labels as ndarray
    :param optimizer_template:
    :param loss_algorithm: the loss algorithm (i.e. BCEWithLogitsLoss(), MSELoss(), etc.)
    :param batch_size: "batch", "online" or the mini-batch size for training
    :param epochs: the number of epochs for training
    :param val_ratio: the ratio of validation to training (i.e. 0.2 = 20% of dataset reserved for training)
    :param patience: the number of epochs without improvement, passing 0 or lower will disable this capability
    :param min_delta: the delta between epochs for early stopping. Ignored if patience < 1
    :param scheduler: learning rate scheduler
    :param silence_output: True/False if we want to display the train params
    :return: tuple(record_tr_loss,record_vl_loss,record_tr_acc,record_vl_acc)
    """

    # Data container for later
    # visual representation
    record_tr_loss = []
    record_vl_loss = []
    record_tr_acc = []
    record_vl_acc = []
    record_grad_norm = []
    epoch_grad_norms = []

    best_vl = float("inf")
    best_state = None
    bad_epochs = 0

    # initialize optimizer
    optimizer = optimizer_template(model.parameters())
    if scheduler_template is not None:
        scheduler = scheduler_template(optimizer)
    else:
        scheduler = None

    # Prepare tensors from TR dataset
    X_tr_t, y_tr_t, X_vl_t, y_vl_t = _prepare_for_torch(
        X.astype(np.float32).to_numpy(), y, val_ratio=val_ratio, seed=seed
    )

    # Build DataLoader (or the batch as it is known in ML)
    if batch_size=="batch":
        batch_size = X_tr_t.shape[0]
    elif batch_size=="online":
        batch_size = 1
    dl_tr, dl_vl = _make_loaders(X_tr_t, y_tr_t, X_vl_t, y_vl_t, batch_size=batch_size)

    if not silence_output:
        _print_train_summary(model, dl_tr,dl_vl,optimizer_template,loss_algorithm, scheduler_template, batch_size, epochs,patience,min_delta)

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
            loss = compute_loss(model, X, y, loss_algorithm)
            loss.backward()
            # Take gradient measurement before update
            epoch_grad_norms.append(gradient_norm(model))
            optimizer.step()

        # Loss estimation
        tr_l = epoch_loss(model, dl_tr, loss_algorithm)
        vl_l = epoch_loss(model, dl_vl, loss_algorithm)

        # Applying learning rate decay
        if scheduler is not None:
            scheduler.step(vl_l)

        # ---- EARLY STOPPING LOGIC ----
        if patience > 0:
            if vl_l < best_vl - min_delta:
                best_vl = vl_l
                bad_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1

            #if epoch % 20 == 0:
            #    print(f"Epoch {epoch:3d} | TR loss: {tr_l:.4f} | VL loss: {vl_l:.4f} | bad: {bad_epochs}/{patience}")

            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best VL loss: {best_vl:.4f})")
                break
        # -----------------------------

        record_tr_loss.append(tr_l)
        record_vl_loss.append(vl_l)

        tr_acc = epoch_accuracy(model, dl_tr)
        vl_acc = epoch_accuracy(model, dl_vl)

        record_tr_acc.append(tr_acc)
        record_vl_acc.append(vl_acc)

        # Gradient mean
        gn = np.mean(epoch_grad_norms)
        record_grad_norm.append(gn)

        # In case early stopping is enabled
        if best_state is not None:
            model.load_state_dict(best_state)

    train_results = {
        "hist_tr": record_tr_loss,
        "hist_vl": record_vl_loss,
        "hist_tr_acc": record_tr_acc,
        "hist_vl_acc": record_vl_acc,
        "hist_grad": record_grad_norm
    }

    return train_results


def run_kfold(untrained_base_model: MLP, X, y, optimizer_template, scheduler_template, loss_function, fold_strategy, inner_train_params: dict):
    """
    Runs K-Fold cross-validation by reusing existing training/evaluation code.
    Returns per-fold histories and final validation accuracies.
    Each fold is trained using the train() function.
    :param untrained_base_model: the untrained MLP model
    :param X: training data
    :param y: training labels
    :param optimizer_template: the weight update algorithm
    :param scheduler_template: the learning rate decay scheduler
    :param loss_function: the chosen loss function (i.e. BCEWithLogitsLoss(), MSELoss(), etc.)
    :param fold_strategy: the fold strategy (i.e. KFold, StratifiedKfold,...)
    :param inner_train_params: the parameters for the inner trainer
    """

    fold_histories = []

    inner_epochs = inner_train_params.get("epochs", DEFAULT_TRAIN_EPOCHS)
    inner_patience = inner_train_params.get("patience", DEFAULT_TRAIN_PATIENCE)
    inner_batch_size = inner_train_params.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE)
    inner_min_delta = inner_train_params.get("min_delta", DEFAULT_TRAIN_DELTA)
    inner_seed = inner_train_params.get("seed", DEFAULT_TRAIN_SEED)

    #_print_train_summary(untrained_base_model, X, y, optimizer_template, loss_function, inner_epochs, inner_patience, inner_min_delta)

    # Fold iteration
    for fold_nr, (tr_idx, vl_idx) in enumerate(fold_strategy.split(X, y)):

        # X, y subset
        X_subset = X.iloc[tr_idx]
        y_subset = y[tr_idx]

        print(f"  ")
        print(f"Perform fold: {fold_nr}, train size: {len(X_subset)}, val size: {len(vl_idx)}")

        # Model cloning
        fold_model = copy.deepcopy(untrained_base_model)

        # Training
        #tr_loss_hist, vl_loss_hist, tr_hist_acc, vl_hist_acc, grad_norm
        train_result = train(
            fold_model, X_subset, y_subset, optimizer_template, loss_function,
            epochs=inner_epochs,
            patience=inner_patience,
            min_delta=inner_min_delta,
            batch_size=inner_batch_size,
            silence_output=False,
            seed=inner_seed,
            scheduler_template=scheduler_template
        )

        hist_tr = train_result["hist_tr"]
        hist_vl = train_result["hist_vl"]
        tr_hist_acc = train_result["hist_tr_acc"]
        vl_hist_acc = train_result["hist_vl_acc"]
        hist_grad = train_result["hist_grad"]

        # Data gathering
        fold_histories.append({
            "tr_loss": np.mean(hist_tr),
            "tr_loss_std": np.std(hist_tr),
            "vl_loss": np.mean(hist_vl),
            "vl_loss_std": np.std(hist_vl),
            "tr_acc": np.mean(tr_hist_acc),
            "tr_acc_std": np.std(tr_hist_acc),
            "vl_acc": np.mean(vl_hist_acc),
            "vl_acc_std": np.std(vl_hist_acc),

            "best_tr_loss": float(min(hist_tr)),
            "best_tr_acc": float(max(tr_hist_acc)),
            "last_tr_loss": float(hist_tr[-1]),
            "last_tr_acc": float(tr_hist_acc[-1]),

            "best_vl_loss": float(min(hist_vl)),
            "best_vl_acc": float(max(vl_hist_acc)),
            "last_vl_loss": float(hist_vl[-1]),
            "last_vl_acc": float(vl_hist_acc[-1]),
        })

    return fold_histories


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
    elif method == "constant":
        nn.init.constant_(m.weight, 0.0)

    else:
        raise ValueError(f"Unknown initialization method: {method}")

    if m.bias is not None:
        nn.init.zeros_(m.bias)


def gradient_norm(model: torch.nn.Module) -> float:
    """
    This function computes the gradient norm of the model parameters,
    in this implementation it corresponds
    :param model:
    :return:
    """

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


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


def regression_mse_adapter() -> OutputAdapter:
    link = lambda z: z                 # identity
    decision = lambda z: z             # per regressione "pred" = valore
    return OutputAdapter(link, decision)


def plot_epoch_loss(hist_tr, hist_vl):
    plt.figure(figsize=(8,4))
    plt.plot(hist_tr, label="TR loss")
    plt.plot(hist_vl, label="VL loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_epoch_accuracy(hist_tr_acc, hist_vl_acc):
    plt.figure(figsize=(8,4))
    plt.plot(hist_tr_acc, label="TR Accuracy")
    plt.plot(hist_vl_acc, label="VL Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_gradient_norm(grad_norms):
    epochs = range(len(grad_norms))

    plt.figure(figsize=(8,4))
    #plt.scatter(epochs, grad_norms, s=2)  # ← punti, non linee
    plt.plot(grad_norms, label="TR loss")
    plt.xlabel("Epoch")
    plt.ylabel("||∇L||")
    plt.title("Gradient norm per epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_gradient_norm_bars(grad_norms, step=10):
    epochs = np.arange(0, len(grad_norms), step)
    values = [grad_norms[i] for i in epochs]

    plt.figure(figsize=(7, 4))
    plt.bar(epochs, values, width=step * 0.6)
    plt.xlabel("Epoch")
    plt.ylabel("||nabla L||")
    plt.title("Gradient norm (one bar every {} epochs)".format(step))
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def _print_train_summary(model, train_dataloader,validation_dataloader,optimizer_template,loss_algorithm,scheduler_template, batch_size, epochs,patience,min_delta):


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
    print(f"Loss algorithm  | {loss_algorithm}")
    print(f"Optimizer       | class: {optimizer_template.func} , params: {optimizer_template.keywords}")
    if scheduler_template is not None:
        print(f"LR Decay        | params: {scheduler_template.keywords}")
    else:
        print(f"LR Decay        | disabled")
    if patience > 0:
        print(f"Early Stopping  | patience: {patience} , minimum delta: {min_delta}")
    else:
        print(f"Early Stopping  | disabled")
    print("-" * 100)
    # print("Model")
    # print(model)
    # print("-" * 100)


def summarize_kfold_table(fold_histories, use="best"):
    """
    Print a compact per-fold table and mean±std summary.

    Parameters
    ----------
    fold_histories : list[dict]
        Each dict contains keys like best_vl_acc, best_vl_loss, last_vl_acc, last_vl_loss, etc.
    use : {"best","last"}
        Which scalar to report as the fold performance.
        - "best": uses best_vl_acc and best_vl_loss
        - "last": uses last_vl_acc and last_vl_loss
    """
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    acc_key = f"{use}_vl_acc"
    loss_key = f"{use}_vl_loss"

    accs = np.array([h[acc_key] for h in fold_histories], dtype=float)
    losses = np.array([h[loss_key] for h in fold_histories], dtype=float)

    # header
    print(f"K-Fold summary ({use.upper()} metrics)")
    print("-" * 54)
    print(f"{'Fold':>4} | {acc_key:>10} | {loss_key:>11} | {'gap_tr-vl(acc)':>14}")
    print("-" * 54)

    for i, h in enumerate(fold_histories, start=1):
        tr_acc = h[f"{use}_tr_acc"]
        vl_acc = h[acc_key]
        gap = tr_acc - vl_acc
        print(f"{i:>4} | {vl_acc:>10.4f} | {h[loss_key]:>11.6f} | {gap:>14.4f}")

    print("-" * 54)
    print(f"MEAN | {accs.mean():>10.4f} | {losses.mean():>11.6f}")
    print(f"STD  | {accs.std():>10.4f} | {losses.std():>11.6f}")
    print("-" * 54)
    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "loss_mean": float(losses.mean()),
        "loss_std": float(losses.std()),
    }


def plot_kfold_bar_acc(fold_histories, use="best"):
    """
    Bar plot of validation accuracy per fold + mean line.

    Parameters
    ----------
    fold_histories : list[dict]
    use : {"best","last"}
    """
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    acc_key = f"{use}_vl_acc"
    acc = np.array([h[acc_key] for h in fold_histories], dtype=float)
    folds = np.arange(1, len(acc) + 1)
    mean_acc = acc.mean()

    plt.figure(figsize=(7, 4))
    plt.bar(folds, acc)
    plt.axhline(mean_acc, linestyle="--", linewidth=2, label=f"Mean = {mean_acc:.3f}")
    plt.xticks(folds)
    plt.ylim(0, 1.0)
    plt.xlabel("Fold")
    plt.ylabel("Validation accuracy")
    plt.title(f"K-Fold validation accuracy per fold ({use})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kfold_bar_regression(
    fold_histories,
    metric="rmse",
    use="best"
):
    """
    Bar plot of validation regression metric per fold + mean line.

    Parameters
    ----------
    fold_histories : list[dict]
        Each dict must contain keys like:
        - "best_vl_rmse", "last_vl_rmse"
        - or "best_vl_mse", "last_vl_mse"
    metric : {"rmse", "mse", "mae"}
        Regression metric to plot.
    use : {"best","last"}
        Whether to use best or last epoch metric.
    """
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    metric = metric.lower()
    if metric not in {"rmse", "mse", "mae"}:
        raise ValueError("metric must be 'rmse', 'mse' or 'mae'")

    key = f"{use}_vl_{metric}"
    values = np.array([h[key] for h in fold_histories], dtype=float)

    folds = np.arange(1, len(values) + 1)
    mean_val = values.mean()

    plt.figure(figsize=(7, 4))
    plt.bar(folds, values)
    plt.axhline(
        mean_val,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.3f}"
    )

    plt.xticks(folds)
    plt.xlabel("Fold")
    plt.ylabel(metric.upper())
    plt.title(f"K-Fold validation {metric.upper()} per fold ({use})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kfold_bar_vl_loss(fold_histories, use="best", ylabel="Validation loss (MSE)"):
    """
    Bar plot of validation loss per fold + mean line.
    Uses keys: best_vl_loss / last_vl_loss.
    """
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    key = f"{use}_vl_loss"
    vals = np.array([h[key] for h in fold_histories], dtype=float)
    folds = np.arange(1, len(vals) + 1)
    mean_val = vals.mean()

    plt.figure(figsize=(7, 4))
    plt.bar(folds, vals)
    plt.axhline(mean_val, linestyle="--", linewidth=2, label=f"Mean = {mean_val:.4f}")
    plt.xticks(folds)
    plt.xlabel("Fold")
    plt.ylabel(ylabel)
    plt.title(f"K-Fold validation loss per fold ({use})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kfold_bar_vl_rmse(fold_histories, use="best"):
    """
    Bar plot of validation RMSE per fold + mean line.
    Assumes vl_loss stored is MSE (from MSELoss).
    """
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    key = f"{use}_vl_loss"
    mse = np.array([h[key] for h in fold_histories], dtype=float)
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


def kfold_regression_table(fold_histories, use="best", derive_rmse=True):
    """
    Build a K-Fold summary table for regression.

    Parameters
    ----------
    fold_histories : list[dict]
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
    if use not in {"best", "last"}:
        raise ValueError("use must be 'best' or 'last'")

    rows = []

    for i, h in enumerate(fold_histories, start=1):
        tr_loss = h[f"{use}_tr_loss"]
        vl_loss = h[f"{use}_vl_loss"]

        row = {
            "Fold": i,
            "TR_MSE": tr_loss,
            "VL_MSE": vl_loss,
            "Gap(TR-VL)": vl_loss - tr_loss
        }

        if derive_rmse:
            row["TR_RMSE"] = np.sqrt(tr_loss)
            row["VL_RMSE"] = np.sqrt(vl_loss)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Fold")

    # Add mean and std rows
    mean_row = df.mean()
    std_row = df.std()

    df.loc["MEAN"] = mean_row
    df.loc["STD"] = std_row

    print(df)

    return df