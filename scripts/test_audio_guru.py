from aglib import AudioGuru

audio_guru = AudioGuru()

path = ".mp3"

mood, genre, tempo = audio_guru(path)

print(mood, genre, tempo)
