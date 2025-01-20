from aglib.models.genre_model import AudioProcessorGenre, GenreModel
import librosa
import numpy as np

processor = AudioProcessorGenre()

X_train, X_test, y_train, y_test = processor.get_data(
    dataset_name="genre_data",
    dataset_path="DATA/",
)


model = GenreModel()
model.load_model()

y, sr = librosa.load(".mp3", mono=True, sr=None)
X = processor.process_data(y, sr=sr)

labels = [
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
]

predicts = []

for features in X:
    predict = model.predict(features, labels)
    predicts.append(predict[0])

unique_values, counts = np.unique(predicts, return_counts=True)

sorted_indices = np.argsort(-counts)
sorted_unique_values = unique_values[sorted_indices]
sorted_counts = counts[sorted_indices]

print(
    f"Распознанный жанр: {sorted_unique_values[0]}, вероятность: {round(sorted_counts[0]/sum(sorted_counts), 2)}"
)
if len(sorted_unique_values) > 1:
    print(
        f"Распознанный жанр: {sorted_unique_values[1]}, вероятность: {round(sorted_counts[1]/sum(sorted_counts), 2)}"
    )
if len(sorted_unique_values) > 2:
    print(
        f"Распознанный жанр: {sorted_unique_values[2]}, вероятность: {round(sorted_counts[2]/sum(sorted_counts), 2)}"
    )
