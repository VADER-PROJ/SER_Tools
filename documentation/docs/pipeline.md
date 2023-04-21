
# SER Pipeline

The audio pipeline for performing SER on a video conference system can be used both in online and offline time, and it contains several stages for properly identifying emotional content from the audio.

The first step of the pipeline is to continuously consume binary audio data of a video conference participant corresponding with a certain duration (defined in the class parameters).

The next step is converting the consumed binary data to an array of floats, and afterward, normalizing the audio signal. The normalization consists of, when necessary, **resampling** the audio to a sampling rate of 16000 Hz and converting the signal to **mono** by averaging samples across the channels.

The third step of the pipeline is to detect voiced speech of the previously consumed second of audio, using the **Silero Voice Activity Detection (VAD)** model (the minimum confidence level associated with the detection of voiced speech is one class parameter). 

Finally, the pipeline consumes a second of audio, it stores the segment if there is enough confidence that it detected voice activity, if it does not pass the threshold and it has previously saved any audio segment, it feeds it to a SER model to predict the emotion in the segment. It is also flexible in terms of duration for the detected segments, as it can be set in configurations the minimum and maximum duration a segment can have.

## Usage Examples

The pipeline class is simple to use, it requires defining a set of parameters of the class, and then it must be fed consequent audio data every with at least 1 second of duration. There is an example of the pipeline showing a real-time progress plot of the detected emotions in the file `real_time_example.ipynb`, and, here is the code without the jupyter notebook plot:

    import pyaudio
    import numpy as np
    from ser_pipeline import SERPipeline

    # create the pipeline
    ser_pipeline = SERPipeline(config_file='config.json')

    while (True):
        # create an audio stream from the user microphone that reads 1 second of data each time
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paFloat32,   # 32 bits
            channels=1,                 # mono
            rate=16000,                 # 16000 Hz
            input=True)

        # feed the pipeline every second
        proba = ser_pipeline.consume(stream.read(16000))
        if proba != None:
            print(f"Emotions Probabilities: {proba}")
            print(f"Recognized emotion: {max(proba, key=proba.get)}\n")


Output:

    Emotions Probabilities: {'anger': 0.1308593, 'happiness': 0.421875, 'sadness': 0.1796875, 'neutral': 0.267578}
    Recognized emotion: happiness

    Emotions Probabilities: {'anger': 0.2753906, 'happiness': 0.394531, 'sadness': 0.109375, 'neutral': 0.2207031}
    Recognized emotion: happiness
