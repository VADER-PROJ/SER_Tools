# Configurations

In addition to manually passing the parameters to both classes, there is also an option to pass a JSON configuration file. An example of this config is present in the file `config.json`:

    "MODELS_DIR": "models" -> Path for the directory where the machine learning models are stored
    "TRADITIONAL_SER": true -> Type of SER model to utilize
    "STRATIFIED": true -> Use SER models resulting from the stratification study
    "FORMAT": "float32" -> Data type of the audio samples fed to the pipeline
    "SAMPLE_RATE": 16000 -> Sample rate of the audio fed to the pipeline
    "NO_CHANNELS": 1 -> Number of audio channels of the audio fed to the pipeline (1 for mono, 2 for stereo)
    "MIN_CONFIDENCE": 0.6 -> Minimum confidence level for voice activity detection
    "MIN_DURATION": 1 -> Minimum duration of speech segments to be classified (in seconds)
    "MAX_DURATION":  -> Maximum duration of speech segments to be classified (in seconds)