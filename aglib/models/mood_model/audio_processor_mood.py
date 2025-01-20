import numpy as np
import torch
import librosa
from ..audio_processor import AudioProcessor
from typing import Union

SAMPLING_RATE = 44100
AUDIO_FRAME_SIZE = 2048
AUDIO_HOP_LENGTH = AUDIO_FRAME_SIZE // 2


class AudioProcessorMood(AudioProcessor):
    """Processes audio data for mood recognition."""

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
            Union[list[np.array], list[torch.Tensor]]: Processed audio features as a tensor if `scale_data` is True,
            otherwise as a NumPy array.
        """
        processed_data_from_audio = self.extract_features(
            wav, sr=sr, n_fft=n_fft, hop_length=hop_length
        )

        if scale_data:
            self.load_scaler()

            scaled_data = []
            for features in processed_data_from_audio:
                scaled_data.append(
                    torch.Tensor(self.scaler.transform(features.reshape(1, -1)))
                )

            return scaled_data

        else:
            return processed_data_from_audio

    def extract_features(
        self,
        wav: np.ndarray,
        sr: float = SAMPLING_RATE,
        n_fft: int = AUDIO_FRAME_SIZE,
        hop_length: int = AUDIO_HOP_LENGTH,
    ) -> list[np.array]:
        """
        Extracts audio features from the provided data.

        Args:
            wav (np.ndarray): Audio data as a NumPy array.
            sr (float): Sampling rate of the audio. Defaults to SAMPLING_RATE.
            n_fft (int): Number of FFT components. Defaults to AUDIO_FRAME_SIZE.
            hop_length (int): Number of samples between frames. Defaults to AUDIO_HOP_LENGTH.

        Returns:
            list[np.array]: Extracted audio features
        """
        audio_features = []
        for y in self._cut(wav, sr):
            result = np.array([])
            zcr = np.mean(
                librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).T, axis=0
            )
            result = np.hstack((result, zcr))

            stft = np.abs(librosa.stft(y))
            chroma_stft = np.mean(
                librosa.feature.chroma_stft(
                    S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length
                ).T,
                axis=0,
            )
            result = np.hstack((result, chroma_stft))

            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
            result = np.hstack((result, mfcc))

            rms = np.mean(librosa.feature.rms(y=y, frame_length=100).T, axis=0)
            result = np.hstack((result, rms))

            mel = np.mean(
                librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length).T,
                axis=0,
            )
            result = np.hstack((result, mel))

            audio_features.append(result)

        return audio_features

    def _cut(self, wav, sr):
        """Cuts audio to 3-second segments"""
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
