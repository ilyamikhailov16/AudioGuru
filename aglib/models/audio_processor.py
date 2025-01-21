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

SAMPLING_RATE = 44100
AUDIO_FRAME_SIZE = 2048
AUDIO_HOP_LENGTH = AUDIO_FRAME_SIZE // 2


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
        """Saves the scaler to the same folder as the processor's class.

        This method saves the current state of the scaler object to a file in the
        same directory as the processor's class. The scaler is saved with a filename
        derived from the processor's path, using the base name followed by
        '_scaler.save'.

        Raises:
            Exception: If there are any issues encountered during the saving process.

        Example:
            >>> processor.save_scaler()
            # This will save the scaler to a file named '<processor_base_name>_scaler.save'
        """
        load_name, extension = self.processor_path.split(".")
        joblib.dump(self.scaler, f"{load_name}_scaler.save")

    def load_scaler(self):
        """Loads the scaler from the same folder as the processor's class.

        This method loads a pre-trained scaler object, which is used to normalize
        or standardize data, from a file located in the same directory as the
        processor's class. The scaler is expected to be saved with a filename
        derived from the processor's path.

        The method assumes that the scaler is saved in a file named with the
        processor's base name followed by '_scaler.save'.

        Raises:
            FileNotFoundError: If the scaler file does not exist at the expected path.
            Exception: For any other issues encountered during the loading process.

        Example:
            >>> processor.load_scaler()
            >>> print(processor.scaler)
            StandardScaler()  # Example output, depending on the scaler type
        """
        load_name, extension = self.processor_path.split(".")
        self.scaler = joblib.load(f"{load_name}_scaler.save")

    def _cut(self, wav, sr) -> list[np.array]:
        """Cuts audio to 3-second segments.

        This method takes an audio waveform and its sampling rate, and divides the
        audio into segments of 3 seconds each. It discards any remaining audio that
        does not fit into a complete 3-second segment.

        Args:
            wav (list or numpy.ndarray): The audio waveform data, represented as a
                list or array of amplitude values.
            sr (int): The sampling rate of the audio, in samples per second.

        Returns:
            list: A list of audio segments, where each segment is a sublist or
            sub-array of the original waveform, representing 3 seconds of audio.

        Example:
            >>> audio_segments = _cut(wav, sr)
            >>> len(audio_segments[0]) == sr * 3  # Each segment is 3 seconds long
            True
        """
        three_seconds_samples = sr * 3
        length_without_residue = len(wav) - (len(wav) % three_seconds_samples)
        return [
            wav[i : i + three_seconds_samples]
            for i in range(
                0,
                (length_without_residue - three_seconds_samples) + 1,
                three_seconds_samples,
            )
        ]
