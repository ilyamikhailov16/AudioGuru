import librosa
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from ..audio_processor import AudioProcessor
from typing import Union
import torch

SAMPLING_RATE = 22050
AUDIO_FRAME_SIZE = 1024
AUDIO_HOP_LENGTH = 512


class AudioProcessorVoice(AudioProcessor):
    """Processes audio data for genre recognition."""

    def process_data(
        self,
        audio: np.ndarray,
        sr: float = SAMPLING_RATE,
        n_fft: int = AUDIO_FRAME_SIZE,
        hop_length: int = AUDIO_HOP_LENGTH,
        scale_data: bool = True,
    ) -> Union[torch.Tensor, np.array]:
        """
        Processes audio data and extracts features.

        Args:
            audio (np.ndarray): Audio data as a NumPy array.
            sr (float): Sampling rate of the audio. Defaults to SAMPLING_RATE.
            n_fft (int): Number of FFT components. Defaults to AUDIO_FRAME_SIZE.
            hop_length (int): Number of samples between frames. Defaults to AUDIO_HOP_LENGTH.
            scale_data (bool): Whether to scale the data. Defaults to True.

        Returns:
            Union[torch.Tensor, np.array]: Processed audio features as a tensor if `scale_data` is True,
            otherwise as a NumPy array.
        """
        processed_data_from_audio = self.extract_features(
            audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )

        if scale_data:
            self.load_scaler()
            return torch.Tensor(
                self.scaler.transform(processed_data_from_audio.reshape(1, -1))
            )
        else:
            return processed_data_from_audio

    def extract_features(
        self,
        data: np.ndarray,
        sr: float = SAMPLING_RATE,
        n_fft: int = AUDIO_FRAME_SIZE,
        hop_length: int = AUDIO_HOP_LENGTH,
    ) -> np.ndarray:
        """
        Extracts audio features from the provided data.

        Args:
            data (np.ndarray): Audio data as a NumPy array.
            sr (float): Sampling rate of the audio. Defaults to SAMPLING_RATE.
            n_fft (int): Number of FFT components. Defaults to AUDIO_FRAME_SIZE.
            hop_length (int): Number of samples between frames. Defaults to AUDIO_HOP_LENGTH.

        Returns:
            np.ndarray: Extracted audio features as a NumPy array with shape (n_features,), 
            where `n_features` is the total number of aggregated feature statistics.
        """
        features = {
            "centroid": librosa.feature.spectral_centroid(
                y=data, sr=sr, n_fft=n_fft, hop_length=hop_length
            ).ravel(),
            "flux": librosa.onset.onset_strength(y=data, sr=sr).ravel(),
            "rmse": librosa.feature.rms(
                y=data, frame_length=n_fft, hop_length=hop_length
            ).ravel(),
            "zcr": librosa.feature.zero_crossing_rate(
                y=data, frame_length=n_fft, hop_length=hop_length
            ).ravel(),
            "contrast": librosa.feature.spectral_contrast(y=data, sr=sr).ravel(),
            "bandwidth": librosa.feature.spectral_bandwidth(
                y=data, sr=sr, n_fft=n_fft, hop_length=hop_length
            ).ravel(),
            "flatness": librosa.feature.spectral_flatness(
                y=data, n_fft=n_fft, hop_length=hop_length
            ).ravel(),
            "rolloff": librosa.feature.spectral_rolloff(
                y=data, sr=sr, n_fft=n_fft, hop_length=hop_length
            ).ravel(),
        }

        mfcc = librosa.feature.mfcc(
            y=data, n_fft=n_fft, hop_length=hop_length, n_mfcc=20
        )
        for idx, v_mfcc in enumerate(mfcc):
            features["mfcc_{}".format(idx)] = v_mfcc.ravel()

        dict_agg_features = self._get_feature_stats(features)
        dict_agg_features["tempo"] = librosa.feature.tempo(
            y=data, sr=sr, hop_length=hop_length
        )[0]

        return np.array(list(dict_agg_features.values()))

    def _get_feature_stats(self, features):
        """
        Calculates statistical features from extracted audio features.

        Args:
            features (dict): A dictionary where keys are feature names and values are their corresponding data.

        Returns:
            dict: A dictionary containing the statistical features (max, min, mean, std, kurtosis, skew).
        """
        result = {}
        for k, v in features.items():
            result["{}_max".format(k)] = np.max(v)
            result["{}_min".format(k)] = np.min(v)
            result["{}_mean".format(k)] = np.mean(v)
            result["{}_std".format(k)] = np.std(v)
            result["{}_kurtosis".format(k)] = kurtosis(v)
            result["{}_skew".format(k)] = skew(v)
        return result
