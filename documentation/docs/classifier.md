# SER Models

The developed models are available through the `SERClassifier` class in the file `ser_classifier.py`. It allows choosing the model developed and trained on the IEMOCAP dataset using a **traditional feature-based SER** and another using a **deep learning-based SER** approach.

This class has built-in methods that allow users to predict emotions from an audio file or directly from an audio vector. The audio must have **16000 Hz** frequency and be **mono** channel. The class also **preprocesses** the audio by **reducing the noise** and **trimming silence** at the beginning and end of the audio before extracting features for classification.

The traditional model constitutes an AdaBoost with Random Forests as the base estimator and utilizes manually extracted audio features from the audio signals to make emotional predictions.

The deep learning model utilizes transfer learning techniques, using a pre-trained ResNet-50 on the ImageNet dataset. This classifier utilizes the image of the audio spectrogram to make classifications.

Additionally, there is the `STRATIFIED` argument of the `SERClassifier` class, which allows selecting the usage of these models trained on limited data that achieved better results, based on a set of conditions that resulted from a study of the training dataset limitations.

## Usage Examples

Here is an example of how to use the classifier class:

    import librosa
    from ser_classifier import SERClassifier

    models_dir, audio_file = "path_to_ml_models_dir", "path_to_audio_file"

    trad_model = SERClassifier(config_file='config.json')

    print("Using the audio file path:")
    print(trad_model.predict(audio_file, is_file=True, return_proba=True))
    print(trad_model.predict(audio_file, is_file=True, return_proba=False))

    y, sr = librosa.load(audio_file, sr=16000)
    print("\nUsing the audio signal vector:")
    print(trad_model.predict(y, is_file=False, return_proba=True))
    print(trad_model.predict(y, is_file=False, return_proba=False))

Output:

    Using the audio file path:
    Probabilities: {'anger': 0.02343750, 'happiness': 0.08203125, 'sadness': 0.08593749, 'neutral': 0.80859375}
    Emotion Label: neutral

    Using the audio signal vector:
    Probabilities: {'anger': 0.02343750, 'happiness': 0.08203125, 'sadness': 0.08593749, 'neutral': 0.80859375}
    Emotion Label: neutral
