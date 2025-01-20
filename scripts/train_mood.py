import torch
import torch.nn as nn
from aglib.models.mood_model import AudioProcessorMood, MoodModel
from aglib.models.plot import plot_graph


if __name__ == "__main__":
    processor = AudioProcessorMood()

    X_train, X_test, y_train, y_test = processor.get_data(
        dataset_name="mood_data",
        dataset_path="DATA/",
    )

    model = MoodModel()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")

    # model.load_model()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=0.001
    )  # lr=1e-5, weight_decay=1e-4
    epochs = 3000

    model, train_losses, test_losses = model.train_model(
        X_train, y_train, X_test, y_test, loss_function, optimizer, epochs
    )

    # model.save_model()

    plot_graph(model, X_test, y_test, train_losses, test_losses)
