import io
import json
import librosa
import numpy as np
import librosa.display
import tensorflow as tf
import noisereduce as nr
import matplotlib.pyplot as plt
from PIL import Image
from joblib import load
from typing import List, Union
from scipy.stats import kurtosis
from keras.utils import img_to_array

class SERClassifier:
    def __init__(
        self,
        config_file: str = None,
        MODELS_DIR: str = None,
        TRADITIONAL_SER: bool = True,
        DEEP_LEARNING_SER: bool = True,
        STRATIFIED: bool = False,
    ) -> None:
        """
        Constructor for SERClassifier class.
        Loads the selected SER classifier model from the corresponding file.

        Args:
            config_file (str): Path for the JSON configuration file.
            MODELS_DIR (str): The path for the directory where the machine learning models are stored.
            TRADITIONAL_SER (bool): Use the traditional model for classying audio.
            DEEP_LEARNING_SER (bool): Use the deep learning model for classying audio.
            STRATIFIED (bool): Set to use the models resulting of the data stratification study.
        """
        if config_file:
            with open(config_file, "r") as f:
                (MODELS_DIR, self.TRADITIONAL_SER, self.DEEP_LEARNING_SER, STRATIFIED,
                 _, _, _, _, _, _) = json.load(f).values()
        else:
            self.TRADITIONAL_SER = TRADITIONAL_SER
            self.DEEP_LEARNING_SER = DEEP_LEARNING_SER

        self.STRATIFIED = STRATIFIED

        if not MODELS_DIR:
            raise Exception(
                "Path for the machine learning models directory is required."
            )

        self.models = {}
        
        if self.TRADITIONAL_SER:
            self.models["traditional"] = load(
                    f"{MODELS_DIR}/{'stratified_' if self.STRATIFIED else ''}traditional_model.pkl"
                )

        if self.DEEP_LEARNING_SER:
            self.models["dl"] = tf.keras.models.load_model(
                    f"{MODELS_DIR}/{'stratified_' if self.STRATIFIED else ''}dl_model.h5"
                )
        
        # If no models were selected, use the traditional model

        if not self.models:
            self.models["traditional"] = load(
                    f"{MODELS_DIR}/{'stratified_' if self.STRATIFIED else ''}traditional_model.pkl"
                )
        

    def spikes(self, data: np.ndarray) -> float:
        """
        Compute the proportion of spikes (values above a threshold) in the given data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            float: Proportion of spikes in the input data.
        """
        if len(data.shape) != 1:
            data = np.concatenate(data)
        mean = np.mean(data)
        std = np.std(data)
        threshold = mean + np.abs(std) * 2 / 100
        num_spikes = 0
        for value in data:
            if value >= threshold:
                num_spikes += 1
        num_spikes = num_spikes / len(data)
        return num_spikes

    def preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply noise reduction and trimming to the given audio signal.

        Args:
            y (np.ndarray): Audio signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            np.ndarray: Preprocessed audio signal.
        """
        y = nr.reduce_noise(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            prop_decrease=0.75,
            time_constant_s=1,
        )
        y, _ = librosa.effects.trim(y, top_db=30)
        return y

    def extract_trad_features(
        self, data: Union[str, np.ndarray], sr: int = 16000
    ) -> List:
        """
        Extract the SER traditional features from the given audio file or signal.

        Args:
            data (Union[str, np.ndarray]): Audio file name or signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            List: List of feature values.
        """
        y = self.preprocess_audio(data, sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        zcr = librosa.feature.zero_crossing_rate(y=y)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        return np.array(
            [
                np.min(zcr),
                self.spikes(zcr),
                np.var(mel_spect),
                self.spikes(mel_spect),
                np.percentile(chroma_stft, 0.25),
                self.spikes(chroma_stft),
                np.mean(spec_bw),
                np.max(spec_bw),
                np.percentile(librosa.feature.rms(y=y), 0.25),
                np.var(mfcc[0]),
                np.var(mfcc[2]),
                np.max(mfcc[4]),
                np.var(mfcc[4]),
                np.median(mfcc[4]),
                self.spikes(mfcc[5]),
                np.percentile(mfcc[6], 0.75),
                np.max(mfcc[6]),
                np.var(mfcc[7]),
                np.sum(mfcc[9]),
                np.max(mfcc[9]),
                np.percentile(mfcc[10], 0.75),
                np.max(mfcc[10]),
                np.sum(mfcc[11]),
                kurtosis(mfcc[11]),
                np.mean(mfcc[12]),
                np.mean(mfcc[14]),
                self.spikes(mfcc[15]),
                kurtosis(mfcc[16]),
                np.mean(mfcc[16]),
                kurtosis(mfcc[17]),
                self.spikes(mfcc[18]),
                np.mean(mfcc[18]),
                np.mean(mfcc[19]),
            ],
            np.float64,
        ).reshape(1, -1)

    def extract_dl_features(
        self, data: Union[str, np.ndarray], sr: int = 16000
    ) -> List:
        """
        Extract the SER deep learning features from the given audio file or signal.

        Args:
            data (Union[str, np.ndarray]): Audio file path or signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            List: Spectrogram image array with (1, 224, 224, 3) shape.
        """
        y = self.preprocess_audio(data, sr)

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max
        )
        librosa.display.specshow(
            spec, sr=sr, hop_length=512, ax=ax, cmap="viridis_r")
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", pad_inches=0, dpi=100)
        buf.seek(0)
        fig.clf()
        plt.close(fig)
        img = Image.open(buf).convert("RGB").resize((224, 224), Image.NEAREST)

        return tf.keras.applications.resnet50.preprocess_input(
            img_to_array(img)
        ).reshape((1, 224, 224, 3))

    def predict(
        self, data: Union[str, np.ndarray], return_proba=True
    ):
        """
        Predicts the emotion label or probabilities for a given audio segment.

        Args:
            data (Union[str, np.ndarray]): Audio file path or a numpy array containing the audio segment.
            return_proba (bool): Whether to return the probabilities instead of the predicted label.

        Returns:
            dict: The predicted discrete emotion label (one of "neutral", "anger", "happiness", "sadness") if `return_proba` is False,
            otherwise a dictionary containing the predicted probabilities for each emotion label, for each model used.
        """

        emotions_detected = {}

        if self.TRADITIONAL_SER:
            emotions_detected[f"{'stratified ' if self.STRATIFIED else ''}traditional"] = self.models["traditional"].predict_proba(
                    self.extract_trad_features(data)
                )[0]

        if self.DEEP_LEARNING_SER:
            emotions_detected[f"{'stratified ' if self.STRATIFIED else ''}deep learning"] = self.models["dl"].predict(
                    self.extract_dl_features(data), verbose=0
                )[0]

        for model, emotions in emotions_detected.items():
            emotions_proba = {
                "anger": emotions[0],
                "happiness": emotions[1],
                "sadness": emotions[2],
                "neutral": emotions[3],
            }

            if not return_proba:
                emotions_detected[model] = max(emotions_proba, key=emotions_proba.get)
            else:
                emotions_detected[model] = emotions_proba

        return emotions_detected
