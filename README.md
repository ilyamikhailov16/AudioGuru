# AudioGuru: Advanced Music Markup

## Description

This is a little library for genre, mood, tempo and instrumental tagging purposes that was made by group of IRIT-RTF students. 
Our markup file solution gives additional information for your tracks, which can be useful in many ways. For example, you can implement a smart searching by combining music tags in your musical service.
In general, it's just a package of models and it's assisting tools. You can freely install it and integrate in your applications. Even if you are not expierienced in machine learning, you can use it by simple interface, that gives easy access to interaction with models

## Table of Contents

- [Installation](#installation)
- [Content](#content)
- [Contributing](#contributing)
- [License](#license)
- [Used Resources](#used_resources)

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/ilyamikhailov16/Audio_Guru.git

# Install dependencies
pip install -r requirements.txt

```

## Content

User interface class defined in aglib.audio_guru. To process a track and get tag predictions just call the process_audio() method.

Models with dependencies are stored in aglib.models package. For features extraction use the AudioProcessor.process_data() method.

We have 3 models:
1) Genre Model. We've used XGboost pretrained model on GZTAN dataset with 5 genres ...
2) Mood Model. A simple FCNN (2000-2000-2000). ReLu activation in hidden layers. Input layer has 162 neurons and output layer has 5 neurons ...
3) Instrumental Model. A simple FCNN (2000-2000-2000). ReLu activation in hidden layers. Input layer have 162 neurons and output ...

And datasets respectively we have used for training:
1) GZTAN and ISMIR2004 (80% training / 20% testing). Before training we preprocessed audio files using these parameters: native serialization rate, mono channel, 16 bit. Each track is divided on 3s long chunks.
2) mood-music-classification (80% training / 20% testing) from Kaggle.
3) NSyth (80% training / 20% testing) from Google Magenta.

You can test our library by running telegram_bot.py, which creates a bot, or by running test_audio_guru.py script.

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

we will place here cites and links later...