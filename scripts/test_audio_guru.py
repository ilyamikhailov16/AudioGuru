from aglib import AudioGuru

audio_guru = AudioGuru()

path = ".mp3"

mood, genre, tempo = audio_guru(path, mode_tag=False)

print(mood, genre, tempo)
