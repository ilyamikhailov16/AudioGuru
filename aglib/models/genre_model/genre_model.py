import torch.nn as nn
from torch import Tensor
from ..model import Model


class GenreModel(Model):
    """Implements a feedforward neural network for genre classification."""

    def __init__(self) -> None:
        """
        Initializes the GenreModel with layers for processing audio features.

        The model consists of four linear layers with ReLU activation functions
        and dropout for regularization.
        """
        super().__init__()

        self.linear1 = nn.Linear(57, 5700)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(5700, 5700)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(5700, 5700)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout()
        self.linear4 = nn.Linear(5700, 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor containing audio features.

        Returns:
            Tensor: Output tensor containing model predictions for each of the 10 genres.
        """
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x = self.linear4(x)

        return x
