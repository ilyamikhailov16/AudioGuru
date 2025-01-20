import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Optional, Union
from abc import ABC, abstractmethod
from inspect import getsourcefile
from os.path import abspath
import joblib

SAMPLING_RATE = 22050
AUDIO_FRAME_SIZE = 1024
AUDIO_HOP_LENGTH = 512


class AudioProcessor(ABC):
    """
    Base class for audio processors of different data type, providing
    common functionalities for audio feature extraction and dataset creating.
    """

    def __init__(self):
        """Initializes AudioProcessor, sets the processor path, scaler, and encoder."""
        self.processor_path = abspath(getsourcefile(self.__class__))
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    @abstractmethod
    def process_data(
        self,
        wav: np.ndarray,
        sr: float = SAMPLING_RATE,
        n_fft: int = AUDIO_FRAME_SIZE,
        hop_length: int = AUDIO_HOP_LENGTH,
        scale_data: bool = True,
    ) -> Union[list[np.array], list[torch.Tensor]]:
        """
        Processes audio data and extracts features.

        Args:
            wav (np.ndarray): Audio data as a NumPy array.
            sr (float): Sampling rate of the audio. Defaults to SAMPLING_RATE.
            n_fft (int): Number of FFT components. Defaults to AUDIO_FRAME_SIZE.
            hop_length (int): Number of samples between frames. Defaults to AUDIO_HOP_LENGTH.
            scale_data (bool): Whether to scale the data. Defaults to True.

        Returns:
            Union[list[np.array], list[torch.Tensor]]: Extracted audio features as a tensor or NumPy array.
        """
        pass

    def get_data(
        self,
        dataset_name: str,
        dataset_path: str,
        shuffle_seed: Optional[int] = 42,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
        """
        Reads and preprocesses dataset from a CSV file.

        Args:
            dataset_name (str): Name of the dataset file (without extension).
            dataset_path (str): Path to the dataset folder.
            shuffle_seed (Optional[int]): Seed for shuffling the data. Defaults to 42.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
            Training and testing data and labels in tensor format.
        """
        data = pd.read_csv(dataset_path + dataset_name + ".csv")

        X = data.drop(["label"], axis=1)
        y = data.pop("label")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=shuffle_seed, test_size=0.2, stratify=y
        )

        y_train = torch.LongTensor(np.array(y_train))
        y_test = torch.LongTensor(np.array(y_test))
        X_train = torch.Tensor(self.scaler.fit_transform(np.array(X_train)))
        X_test = torch.Tensor(self.scaler.transform(np.array(X_test)))

        self.save_scaler()

        return (X_train, X_test, y_train, y_test)

    @abstractmethod
    def extract_features(
        self,
        wav: np.ndarray,
        sr: float = SAMPLING_RATE,
        n_fft: int = AUDIO_FRAME_SIZE,
        hop_length: int = AUDIO_HOP_LENGTH,
    ) -> list[np.array]:
        """
        Extracts features from audio data.

        Args:
            wav (np.ndarray): Audio data as a NumPy array.
            sr (float): Sampling rate of the audio. Defaults to SAMPLING_RATE.
            n_fft (int): Number of FFT components. Defaults to AUDIO_FRAME_SIZE.
            hop_length (int): Number of samples between frames. Defaults to AUDIO_HOP_LENGTH.

        Returns:
            list[np.array]: Extracted audio features as a NumPy array.
        """
        pass

    def save_scaler(self):
        """Saves the scaler to the same folder as the processor's class."""
        load_name, extension = self.processor_path.split(".")
        joblib.dump(self.scaler, f"{load_name}_scaler.save")

    def load_scaler(self):
        """Loads the scaler from the same folder as the processor's class."""
        load_name, extension = self.processor_path.split(".")
        self.scaler = joblib.load(f"{load_name}_scaler.save")
