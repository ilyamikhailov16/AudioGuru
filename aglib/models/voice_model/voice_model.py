import torch.nn as nn
from torch import Tensor
from ..model import Model


class VoiceModel(Model):
    """
    A neural network model for voice-related tasks.

    This model consists of multiple fully connected layers with ReLU activations
    and dropout for regularization. It takes an input tensor of shape (batch_size, 162)
    and produces an output tensor of shape (batch_size, 5) representing class scores.

    Inherits from the `Model` class.
    """
    def __init__(self) -> None:
        """
        Initializes an instance of the VoiceModel class.

        This constructor sets up the model architecture, including the linear layers,
        activation functions, and dropout layers.

        Attributes:
            linear1 (nn.Linear): The first linear layer transforming input from 162 to 2000 dimensions.
            act1 (nn.ReLU): The activation function after the first linear layer.
            dropout1 (nn.Dropout): The dropout layer for regularization after the first activation.
            linear2 (nn.Linear): The second linear layer, maintaining a dimensionality of 2000.
            act2 (nn.ReLU): The activation function after the second linear layer.
            dropout2 (nn.Dropout): The dropout layer for regularization after the second activation.
            linear3 (nn.Linear): The third linear layer, maintaining a dimensionality of 2000.
            act3 (nn.ReLU): The activation function after the third linear layer.
            dropout3 (nn.Dropout): The dropout layer for regularization after the third activation.
            linear4 (nn.Linear): The final linear layer reducing dimensions from 2000 to 5 outputs.
        """
        super().__init__()

        self.linear1 = nn.Linear(162, 2000)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(2000, 2000)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(2000, 2000)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout()
        self.linear4 = nn.Linear(2000, 5)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 162).

        Returns:
            Tensor: Output tensor of shape (batch_size, 5) representing predicted class scores.
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
