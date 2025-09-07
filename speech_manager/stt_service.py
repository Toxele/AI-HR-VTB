import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperSTT:
    def __init__(self, model_name: str = "openai/whisper-large-v3-turbo"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def transcribe_audio(self, audio_path: str, language: str = None, task: str = "transcribe") -> str:
        """
        Транскрибирует аудиофайл в текст с возможностью указания языка и задачи
        Args:
            audio_path: путь к аудиофайлу
            language: язык для транскрипции (None для автоопределения)
            task: "transcribe" для транскрипции или "translate" для перевода на английский
        Returns:
            распознанный текст
        """
        try:
            # Загрузка и препроцессинг аудио
            audio, sr = librosa.load(audio_path, sr=16000)
            input_features = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features.to(self.device)

            # Параметры генерации
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
            if task:
                generate_kwargs["task"] = task

            # Генерация транскрипции
            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    input_features,
                    **generate_kwargs
                )

            # Декодирование результата
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )

            return transcription[0]

        except Exception as e:
            raise Exception(f"Ошибка транскрибации: {str(e)}")

    def transcribe_audio_buffer(self, audio_buffer: np.ndarray, sample_rate: int = 16000,
                                language: str = None, task: str = "transcribe") -> str:
        """
        Транскрибирует аудио из буфера с возможностью указания языка и задачи
        Args:
            audio_buffer: numpy array с аудиоданными
            sample_rate: частота дискретизации
            language: язык для транскрипции (None для автоопределения)
            task: "transcribe" для транскрипции или "translate" для перевода на английский
        Returns:
            распознанный текст
        """
        try:
            # Препроцессинг аудио
            input_features = self.processor(
                audio_buffer,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)

            # Параметры генерации
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
            if task:
                generate_kwargs["task"] = task

            # Генерация транскрипции
            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    input_features,
                    **generate_kwargs
                )

            # Декодирование результата
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )

            return transcription[0]

        except Exception as e:
            raise Exception(f"Ошибка транскрибации из буфера: {str(e)}")