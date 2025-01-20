import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Self
from torcheval.metrics import MulticlassAccuracy
from .early_stopping import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from inspect import getsourcefile
from os.path import abspath


class Model(nn.Module):
    """
    Base class for machine learning models, providing common functionalities
    for prediction, training, and model saving/loading.
    """

    def __init__(self):
        """
        Initializes an instance of the Model class.

        This constructor sets up the basic structure for the machine learning model by
        calling the constructor of the parent nn.Module class and defining the model path.

        Attributes:
            model_path (str): Absolute path to the model file, used for saving
                            and loading the model's state.

        Example:
            model = Model()
            # Creates a model that is ready for further configuration and training.

        Note:
            This class serves as a base class. Specific model architectures should
            inherit from this Model class and define their own architecture in the
            `forward` method.
        """
        super().__init__()
        self.model_path = abspath(getsourcefile(self.__class__))

    def predict(self, X: torch.Tensor, labels: tuple[str]) -> str:
        """
        Predict the label based on the provided features.

        Args:
            X (torch.Tensor): Feature data for prediction.
            labels (tuple[str]): List of possible labels.

        Returns:
            str: Predicted label.
        """
        self.eval()

        y_pred = self(X)
        y_pred_prob = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred_classes = torch.argmax(y_pred_prob, dim=1)

        encoder = LabelEncoder()
        encoder.fit(labels)
        encoder.classes_ = np.array(labels)
        return encoder.inverse_transform(y_pred_classes)

    def save_model(self) -> None:
        """
        Save the model to the same folder as the model's class with the same name.
        """
        save_name, extension = self.model_path.split(".")
        torch.save(self.state_dict(), f"{save_name}.pt")

    def load_model(self) -> None:
        """
        Load the model from the same folder as the model's class with the same name.
        """
        load_name, extension = self.model_path.split(".")
        self.load_state_dict(
            torch.load(
                f"{load_name}.pt",
                weights_only=True,
            ),
        )

    def compute_accuracy(
        self, X_test: torch.Tensor, y_test: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute and return the model's accuracy on the test dataset.

        Args:
            X_test (torch.Tensor): Feature data for testing.
            y_test (torch.LongTensor): True labels for testing.

        Returns:
            torch.Tensor: Computed accuracy.
        """
        self.eval()
        
        metric = MulticlassAccuracy()
        with torch.no_grad():
            y_pred = self(X_test)
            y_pred_prob = torch.nn.functional.softmax(y_pred, dim=1)
            y_pred_classes = torch.argmax(y_pred_prob, dim=1)
            metric.update(y_pred_classes, y_test)

        return metric.compute()

    def train_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.LongTensor,
        X_test: torch.Tensor,
        y_test: torch.LongTensor,
        loss_function: nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        epochs: int,
    ) -> tuple[Self, list[float], list[float]]:
        """
        Train the model by adjusting its weights and biases to minimize the loss.

        Args:
            X_train (torch.Tensor): Training feature data.
            y_train (torch.LongTensor): Training labels.
            X_test (torch.Tensor): Validation feature data.
            y_test (torch.LongTensor): Validation labels.
            loss_function (nn.CrossEntropyLoss): Loss function for training.
            optimizer (torch.optim.Adam): Optimizer for weight updates.
            epochs (int): Number of training epochs.

        Returns:
            tuple[Self, list[float], list[float]]: Tuple containing the trained model, training losses,
            and validation losses.
        """
        train_losses = []
        test_losses = []
        early_stopping = EarlyStopping(patience=10, delta=0.0001)

        for epoch in tqdm(range(epochs)):
            self.train()

            optimizer.zero_grad()

            y_pred = self(X_train)
            train_loss = loss_function(y_pred, y_train)

            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

            self.eval()
            with torch.no_grad():
                y_pred = self(X_test)
                test_loss = loss_function(y_pred, y_test)
                test_losses.append(test_loss.item())

            print(
                f"Epoch: {epoch}, Train Loss: {train_loss.item():.4f}, \
                Test Loss: {test_loss.item():.4f}"
            )

            early_stopping(test_loss.item(), self)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        early_stopping.load_best_model(self)

        return self, train_losses, test_losses
