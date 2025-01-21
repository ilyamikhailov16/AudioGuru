# AudioGuru: Advanced Music Markup

## Description

This is a little library for genre, mood, tempo and instrumental tagging purposes that was made by group of IRIT-RTF students. 
Our markup file solution gives additional information for your tracks, which can be useful in many ways. For example, you can implement a smart searching by combining music tags in your musical service.
In general, it's just a package of models and it's assisting tools. You can freely install it and integrate in your applications. Even if you are not expierienced in machine learning, you can use it by simple interface, that gives easy access to interaction with models.
If you want to dive into code, our documentation might help you https://ganjamember.github.io/audio-guru-documentation/index.html.

## Table of Contents

- [Installation](#installation)
- [Content](#content)
- [Contributing](#contributing)
- [License](#license)
- [Used Materials](#used_materials)

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/ilyamikhailov16/Audio_Guru.git

# Install dependencies
pip install -r requirements.txt

```

## Content

User interface class defined in aglib.audio_guru. To process a track and get tag predictions just import the AudioGuru class and call the process_audio() method of an instance. You can also test our library by running telegram_bot.py, which launcs a bot (your own TG token is required), or by running test_audio_guru.py. Both of them stored in the scripts folder.

Models with dependencies are stored in aglib.models package. You can test and train them separately by tools we have stored in the scripts folder. For features extraction use the AudioProcessor.process_data() method. For getting the data from datasets use the AudioProcessor.get_data() method.

Trained models and trained scalers are saved as .pt and .save files in their models packages. To load them use Model.load_model() and AudioProcessor.load_scaler() in your scripts.

We have 3 models and 3 audio processors for them:
1) Genre Model.A simple FCNN (5700-5700-5700) with Dropout. ReLu activation in hidden layers. Input layer has 169 inputs and output layer has 10 outputs. For features extraction use an AudioProcessorGenre class instance
2) Mood Model. A simple FCNN (2000-2000-2000) with Dropout. ReLu activation in hidden layers. Input layer has 162 inputs and output layer has 5 outputs. For features extraction use an AudioProcessorMood class instance
3) Instrumental Model. A simple FCNN (2000-2000-2000) with Dropout. ReLu activation in hidden layers. Input layer has 57 inputs and output layer has 2 outputs. For features extraction use an AudioProcessorVoice class instance

## Contributing

Guidelines for contributing to the project.

1) Fork the repository.
2) Create a new branch (git checkout -b new_branch_name).
3) Make your changes.
4) Commit your changes (git commit -m 'Add a feature').
5) Push to the branch (git push origin new_branch_name).
6) Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Used Materials

We have used pytorch, scikit-learn and librosa libraries for DNN training and audio preprocessing.
We have used Sphinx for the documentation creation and Github Pages for a web-site hosting.

The datasets we have used for training (they are stored in the DATA folder):
1) GZTAN (80% training / 20% testing). Before training we preprocessed audio files using these parameters: native serialization rate, mono channel, 32 bit. Each track is divided into fragments 3s in length.
2) mood-music-classification (80% training / 20% testing) from Kaggle. For audio preprocessing we used KaggleX: Recognizing Emotions in Music, by Diksha Srivastava
3) NSyth (80% training / 20% testing) from Google Magenta.
Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck,
Karen Simonyan, and Mohammad Norouzi. "Neural Audio Synthesis of Musical Notes
with WaveNet Autoencoders." 2017.

