import json
import torch
import librosa
import numpy as np
from typing import Optional, Dict, Union
from ser_classifier import SERClassifier


class SERPipeline:
    def __init__(
        self,
        config_file: str = None,
        MODELS_DIR: str = None,
        TRADITIONAL_SER: bool = True,
        DEEP_LEARNING_SER=True,
        STRATIFIED: bool = False,
        FORMAT: str = "float32",
        SAMPLE_RATE: int = 16000,
        NO_CHANNELS: int = 1,
        MIN_CONFIDENCE: float = 0.6,
        MIN_DURATION: float = 1,
        MAX_DURATION: float = 6,
        IS_FILE: bool = False,
        RETURN_PROBABILITIES: bool = True,
    ) -> None:
        """
        Initializes an instance of SERPipeline.

        Args:
            config_file (str): Path for the JSON configuration file.
            MODELS_DIR (float): Path for the directory where the machine learning models are stored
            TRADITIONAL_SER (float): Use the traditional classifier
            DEEP_LEARNING_SER (float): Use the deep learning classifier
            STRATIFIED (float): Use SER models resulting of the stratification study
            FORMAT (str): Data type of audio samples (default: 'float32').
            SAMPLE_RATE (int): Sample rate of audio (in Hz) (default: 16000).
            NO_CHANNELS (int): Number of audio channels (1 for mono, 2 for stereo) (default: 1).
            MIN_CONFIDENCE (float): Minimum confidence level for voice activity detection (default: 0.6).
            MIN_DURATION (float): Minimum duration of speech segments (in seconds) (default: 1).
            MAX_DURATION (float): Maximum duration of speech segments (in seconds) (default: 6).
            IS_FILE (bool): Wether the audio consumed is from a file or not (default: False).
            RETURN_PROBABILITIES (bool): Wether to return the predicted emotion probabilities (default: True).
        """

        # Use the parameters in a configurations file
        if config_file:
            with open(config_file, "r") as f:
                (
                    self.MODELS_DIR,
                    self.TRADITIONAL_SER,
                    self.DEEP_LEARNING_SER,
                    self.STRATIFIED,
                    self.FORMAT,
                    self.SAMPLE_RATE,
                    self.NO_CHANNELS,
                    self.MIN_CONFIDENCE,
                    self.MIN_DURATION,
                    self.MAX_DURATION,
                    self.IS_FILE,
                    self.RETURN_PROBABILITIES
                ) = json.load(f).values()
        else:
            (
                self.MODELS_DIR,
                self.TRADITIONAL_SER,
                self.DEEP_LEARNING_SER,
                self.STRATIFIED,
                self.FORMAT,
                self.SAMPLE_RATE,
                self.NO_CHANNELS,
                self.MIN_CONFIDENCE,
                self.MIN_DURATION,
                self.MAX_DURATION,
                self.IS_FILE,
                self.RETURN_PROBABILITIES
            ) = (
                MODELS_DIR,
                TRADITIONAL_SER,
                DEEP_LEARNING_SER,
                STRATIFIED,
                FORMAT,
                SAMPLE_RATE,
                NO_CHANNELS,
                MIN_CONFIDENCE,
                MIN_DURATION,
                MAX_DURATION,
                IS_FILE,
                RETURN_PROBABILITIES
            )

        # In case both classifiers weren't selected,
        # select the traditional model
        if not self.TRADITIONAL_SER and not self.DEEP_LEARNING_SER:
            traditional_ser = True

        # Initialize variables for segmentation

        (self.current_y, self.prev_start, self.prev_end) = (None, None, None)
        (self.start, self.step) = (0, 1)

        # Load VAD and SER models

        (self.vad_model, _) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )

        if not self.MODELS_DIR:
            raise Exception(
                "Path for the machine learning models directory is required."
            )

        self.classifier_model = SERClassifier(
            MODELS_DIR=self.MODELS_DIR,
            TRADITIONAL_SER=self.TRADITIONAL_SER,
            DEEP_LEARNING_SER=self.DEEP_LEARNING_SER,
            STRATIFIED=self.STRATIFIED
        )

    def process_bytes(self, y: Union[bytes, str]) -> np.ndarray:
        """
        Converts raw audio bytes to a numpy array.

        Args:
            y (bytes or str): Raw audio data in bytes or file name with audio data.

        Returns:
            numpy.ndarray: Numpy array of audio data.
        """
        # Read data from file
        if self.IS_FILE:
            y, self.SAMPLE_RATE = librosa.load(y, sr=16000)

        # Convert bytes to numpy array
        y = np.frombuffer(y, self.FORMAT)
        if self.FORMAT == "int32":
            abs_max = np.abs(y).max()
            y = y.astype("float32")
            if abs_max > 0:
                y *= 1 / abs_max
            y = y.squeeze()
        elif self.FORMAT != "float32":
            y = y.astype("float32")
            
        return y

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Resamples and converts audio to mono if necessary.

        Args:
            y (numpy.ndarray): Numpy array of audio data.

        Returns:
            numpy.ndarray: Resampled and mono audio data.
        """

        # Resample and convert to mono if necessary

        if self.SAMPLE_RATE != 16000:
            y = librosa.resample(y, self.SAMPLE_RATE, 16000)
        if self.NO_CHANNELS != 1:
            y = librosa.to_mono(y)
        return torch.from_numpy(np.array(y))

    def consume(self, binary_audio: bytes) -> Optional[Dict[str, float]]:
        """
        Processes a segment of raw audio data and returns the predicted emotion probabilities.

        Args:
            binary_audio (bytes): Raw audio data in bytes.

        Returns:
            dict or None: A dictionary of emotion probabilities, or None if not enough audio data has been processed.
        """

        # Initialize emotion probabilities

        emotion_prob = None

        # Update start and end times

        self.start += self.step
        end = self.start + self.step

        # Convert binary audio to numpy array

        y = self.process_bytes(binary_audio)

        # Normalize audio

        y = self.normalize_audio(y)

        if self.IS_FILE:
            emotion_prob = self.classifier_model.predict(
                    y, return_proba=self.RETURN_PROBABILITIES
                )
            return emotion_prob

        # Check if given input audio chunk is too short

        if self.SAMPLE_RATE / y.shape[0] > 31.25:
            if self.prev_end - self.prev_start >= self.MIN_DURATION:
                emotion_prob = self.classifier_model.predict(
                    self.current_y, return_proba=self.RETURN_PROBABILITIES
                )
            (self.prev_start, self.prev_end, self.current_y) = (None, None, None)
            return emotion_prob

        # Perform voice activity detection with SileroVAD model

        confidence = self.vad_model(y, self.SAMPLE_RATE).item()

        if confidence >= self.MIN_CONFIDENCE:

            # If confident voiced speech detected, add to current segment

            if self.prev_end == None:
                self.current_y = y
                (self.prev_start, self.prev_end) = (self.start, end)
            else:
                self.current_y = np.append(self.current_y, y)
                self.prev_end = end

                # If current segment exceeds or equals the maximum duration,
                # classify it

                if self.prev_end - self.prev_start >= self.MAX_DURATION:
                    emotion_prob = self.classifier_model.predict(
                        self.current_y, return_proba=self.RETURN_PROBABILITIES
                    )
                    (self.prev_start, self.prev_end, self.current_y) = (
                        None,
                        None,
                        None,
                    )
        elif self.prev_end:

            # If voiced speech stops and
            # the previous segment duration exceeds or equals minimum duration,
            # classify the current segment

            if self.prev_end - self.prev_start >= self.MIN_DURATION:
                emotion_prob = self.classifier_model.predict(
                    self.current_y, return_proba=self.RETURN_PROBABILITIES
                )
            (self.prev_start, self.prev_end, self.current_y) = (None, None, None)

        # Return emotion probabilities
        
        return emotion_prob
