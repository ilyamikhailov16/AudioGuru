from .models.mood_model import MoodModel, AudioProcessorMood
from .models.genre_model import GenreModel, AudioProcessorGenre
from .models.voice_model import VoiceModel, AudioProcessorVoice
import numpy as np
from typing import Union
from numpy import ndarray
import librosa


class AudioGuru:
    """
    A class to process audio files and predict their mood and genre.

    Attributes:
        models (list): A list of models for mood and genre prediction.
        audio_processors (list): A list of audio processors corresponding to the models.
    """

    def __init__(self) -> None:
        """Initializes AudioGuru, loads models, and sets up audio processors."""
        self.models = [MoodModel(), GenreModel()]  # VoiceModel()
        self.audio_processors = [
            AudioProcessorMood(),
            AudioProcessorGenre(),
        ]  # AudioProcessorVoice()
        self._set_up_models()

    def _set_up_models(self) -> None:
        """Loads the machine learning models."""
        for model in self.models:
            model.load_model()

    def get_audio_tempo(self, wav: ndarray, sr: int) -> str:
        """Calculates the tempo of the audio and classifies it.

        Args:
            wav (ndarray audio data as a NumPy array.
            sr (int): Sample rate of the audio.

        Returns:
            str: Classification of the tempo as one of
                "very slow", "slow", "medium", "fast", or "very fast".
        """
        tempo = librosa.feature.tempo(y=wav, sr=sr)[0]

        if tempo <= 60:
            return "very slow"
        elif tempo > 60 and tempo <= 100:
            return "slow"
        elif tempo > 100 and tempo <= 120:
            return "medium"
        elif tempo > 120 and tempo <= 160:
            return "fast"
        elif tempo > 160:
            return "very fast"

    def process_audio(
        self, audio_path: str, labels: tuple[tuple[str]]
    ) -> list[list[tuple], list[tuple], str]:
        """Processes the audio file to predict its tags.

        Args:
            audio_path (str): The file path of the audio to be processed.
            labels (tuple[tuple[str]]): A tuple of mood and genre labels.

        Returns:
            list[list[tuple], list[tuple], str]: A list of predicted tags.
        """
        tags = []

        wav, sr = librosa.load(audio_path, mono=True, sr=None)

        for model, audio_processor, label in zip(
            self.models, self.audio_processors, labels
        ):
            audio_features = audio_processor.process_data(wav, sr=sr)

            predicts = []

            for features in audio_features:
                predict = model.predict(features, label)
                predicts.append(predict[0])

            unique_values, counts = np.unique(predicts, return_counts=True)

            sorted_indices = np.argsort(-counts)
            sorted_unique_values = unique_values[sorted_indices]
            sorted_counts = counts[sorted_indices]

            sum_of_counts = sum(sorted_counts)
            tag = []

            for i in range(len(sorted_unique_values)):
                tag.append(
                    (
                        sorted_unique_values[i],
                        round(sorted_counts[i] / sum_of_counts, 2),
                    )
                )

            tags.append(tag)

        tempo = self.get_audio_tempo(wav, sr)
        tags.append(tempo)

        return tags

    def __call__(
        self, path, mode_tag: bool = True
    ) -> Union[tuple[str], tuple[list[tuple], list[tuple], str]]:
        """Predicts mood, genre, and tempo of an audio file.

        This method processes an audio file to predict its mood, genre, and tempo.
        It can return either a simplified output with the top predictions or detailed
        lists of predictions based on the `mode_tag` parameter.

        Args:
            path (str): The file path to the audio file to be analyzed.
            mode_tag (bool, optional): If True, returns only the top mood, genre, and
                tempo prediction. If False, returns detailed lists of predictions.
                Defaults to True.

        Returns:
            Union[tuple[str], tuple[list[tuple], list[tuple], str]]:
            - If `mode_tag` is True, returns a tuple of the top mood, genre, and tempo
            as strings.
            - If `mode_tag` is False, returns a tuple containing:
            - A list of tuples for mood predictions, each tuple containing a mood
                and its associated score.
            - A list of tuples for genre predictions, each tuple containing a genre
                and its associated score.
            - A string representing the tempo.

        Example:
            >>> mood, genre, tempo = __call__('path/to/audio/file.wav')
            >>> print(mood, genre, tempo)
            'happy' 'pop' '120 BPM'
        """
        labels = (
            ("aggressive", "dramatic", "happy", "romantic", "sad"),
            (
                "blues",
                "classical",
                "country",
                "disco",
                "hiphop",
                "jazz",
                "metal",
                "pop",
                "reggae",
                "rock",
            ),
        )
        mood, genre, tempo = self.process_audio(path, labels)
        if mode_tag:
            return mood[0][0], genre[0][0], tempo
        return mood, genre, tempo
