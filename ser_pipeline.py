import json
import torch
import librosa
import numpy as np
from typing import Optional, Dict
from ser_classifier import SERClassifier


class SERPipeline:
    def __init__(
        self,
        config_file: str = None,
        MODELS_DIR: str = None,
        TRADITIONAL_SER: bool = True,
        STRATIFIED: bool = False,
        FORMAT: str = "float32",
        SAMPLE_RATE: int = 16000,
        NO_CHANNELS: int = 1,
        MIN_CONFIDENCE: float = 0.6,
        MIN_DURATION: float = 1,
        MAX_DURATION: float = 6,
    ) -> None:
        """
        Initializes an instance of SERPipeline.

        Args:
            config_file (str): Path for the JSON configuration file.
            MODELS_DIR (float): Path for the directory where the machine learning models are stored
            TRADITIONAL_SER (float): Type of SER model to utilize
            STRATIFIED (float): Use SER models resulting of the stratification study
            FORMAT (str): Data type of audio samples (default: 'float32').
            SAMPLE_RATE (int): Sample rate of audio (in Hz) (default: 16000).
            NO_CHANNELS (int): Number of audio channels (1 for mono, 2 for stereo) (default: 1).
            MIN_CONFIDENCE (float): Minimum confidence level for voice activity detection (default: 0.6).
            MIN_DURATION (float): Minimum duration of speech segments (in seconds) (default: 1).
            MAX_DURATION (float): Maximum duration of speech segments (in seconds) (default: 6).
        """

        # Use the parameters in a configurations file
        if config_file:
            with open(config_file, "r") as f:
                (
                    self.MODELS_DIR,
                    self.TRADITIONAL_SER,
                    self.STRATIFIED,
                    self.FORMAT,
                    self.SAMPLE_RATE,
                    self.NO_CHANNELS,
                    self.MIN_CONFIDENCE,
                    self.MIN_DURATION,
                    self.MAX_DURATION,
                ) = json.load(f).values()
        else:
            (
                self.MODELS_DIR,
                self.TRADITIONAL_SER,
                self.STRATIFIED,
                self.FORMAT,
                self.SAMPLE_RATE,
                self.NO_CHANNELS,
                self.MIN_CONFIDENCE,
                self.MIN_DURATION,
                self.MAX_DURATION,
            ) = (
                MODELS_DIR,
                TRADITIONAL_SER,
                STRATIFIED,
                FORMAT,
                SAMPLE_RATE,
                NO_CHANNELS,
                MIN_CONFIDENCE,
                MIN_DURATION,
                MAX_DURATION,
            )

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
            MODELS_DIR=self.MODELS_DIR, TRADITIONAL_SER=self.TRADITIONAL_SER, STRATIFIED=self.STRATIFIED
        )

    def process_bytes(self, y: bytes) -> np.ndarray:
        """
        Converts raw audio bytes to a numpy array.

        Args:
            y (bytes): Raw audio data in bytes.

        Returns:
            numpy.ndarray: Numpy array of audio data.
        """

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

        # Check if given input audio chunk is too short

        if self.SAMPLE_RATE / y.shape[0] > 31.25:
            if self.prev_end - self.prev_start >= self.MIN_DURATION:
                emotion_prob = self.classifier_model.predict(
                    self.current_y, is_file=False, return_proba=True
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
                        self.current_y, is_file=False, return_proba=True
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
                    self.current_y, is_file=False, return_proba=True
                )
            (self.prev_start, self.prev_end, self.current_y) = (None, None, None)

        # Return emotion probabilities

        return emotion_prob
