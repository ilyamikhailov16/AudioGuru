from matplotlib import pyplot as plt
from .model import Model
from torch import Tensor, LongTensor


def plot_graph(
    model: Model,
    X_test: Tensor,
    y_test: LongTensor,
    train_losses: list[float],
    test_losses: list[float],
) -> None:
    """
    Display the training state of the model on training and testing data
    as a function of epoch.

    Args:
        model (Model): The model instance to evaluate.
        X_test (torch.Tensor): Test feature data.
        y_test (torch.LongTensor): True labels for the test data.
        train_losses (list[float]): List of training loss values over epochs.
        test_losses (list[float]): List of testing loss values over epochs.

    Returns:
        None
    """
    print(f"Точность: {model.compute_accuracy(X_test, y_test):.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train loss", color="red")
    plt.plot(test_losses, label="Test loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.legend()
    plt.show()
