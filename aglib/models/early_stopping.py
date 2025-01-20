from torch import nn


class EarlyStopping:
    """
    A utility class for early stopping during model training.
    Monitors validation loss to halt training when the model stops improving.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, delta: float = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss: int, model: nn.Module) -> None:
        """
        Update the early stopping criteria based on the validation loss and the current model state.

        Args:
            val_loss (float): The current validation loss.
            model (nn.Module): The model instance to store the state of.

        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model: nn.Module) -> None:
        """
        Load the best model state into the specified model.

        Args:
            model (nn.Module): The model instance to load the state into.

        Returns:
            None
        """
        model.load_state_dict(self.best_model_state)
