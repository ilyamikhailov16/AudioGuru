import librosa
import numpy as np
from ..audio_processor import AudioProcessor
from typing import Union
import torch

SAMPLING_RATE = 44100


class AudioProcessorGenre(AudioProcessor):
    """Processes audio data for genre recognition."""

    def process_data(
        self,
        wav: np.ndarray,
        sr: float = SAMPLING_RATE,
        scale_data: bool = True,
    ) -> Union[list[np.array], list[torch.Tensor]]:
        """
        Processes audio data and extracts features.

        Args:
            wav (np.ndarray): Audio data as a NumPy array.
            sr (float): Sampling rate of the audio. Defaults to SAMPLING_RATE.
            scale_data (bool): Whether to scale the data. Defaults to True.

        Returns:
            Union[list[np.array], list[torch.Tensor]]: Processed audio features as a tensor if `scale_data` is True,
            otherwise as a NumPy array.
        """
        processed_data_from_audio = self.extract_features(wav, sr=sr)

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
    ) -> list[np.array]:
        """
        Extracts audio features from the provided data.

        Args:
            wav (np.ndarray): Audio data as a NumPy array.
            sr (float): Sampling rate of the audio. Defaults to SAMPLING_RATE.

        Returns:
            list[np.array]: Extracted audio features
        """
        audio_features = []
        for y in self._cut(wav, sr):
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            harmony, perceptr = librosa.effects.harmonic(y), librosa.effects.percussive(
                y
            )
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            features = [
                np.mean(chroma_stft),
                np.var(chroma_stft),
                np.mean(rms),
                np.var(rms),
                np.mean(spec_cent),
                np.var(spec_cent),
                np.mean(spec_bw),
                np.var(spec_bw),
                np.mean(rolloff),
                np.var(rolloff),
                np.mean(zcr),
                np.var(zcr),
                np.mean(harmony),
                np.var(harmony),
                np.mean(perceptr),
                np.var(perceptr),
                float(tempo),
            ]

            for coeff in mfcc:
                features.append(np.mean(coeff))
                features.append(np.var(coeff))

            audio_features.append(np.array(features))

        return audio_features
