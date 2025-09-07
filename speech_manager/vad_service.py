import torch
import numpy as np
from collections import deque


class SileroVAD:
    def __init__(self, threshold: float = 0.5, window_size: int = 512):
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.threshold = threshold
        self.window_size = window_size
        self.sample_rate = 16000
        self.speech_buffer = deque(maxlen=100)  # Хранит последние 100 результатов VAD

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Определяет, содержит ли аудиочанк речь
        Args:
            audio_chunk: аудиоданные в формате numpy array (16kHz, float32)
        Returns:
            True если обнаружена речь, иначе False
        """
        try:
            audio_tensor = torch.from_numpy(audio_chunk).float()
            confidence = self.model(audio_tensor, self.sample_rate).item()

            # Добавляем в буфер и принимаем решение на основе скользящего среднего
            self.speech_buffer.append(confidence > self.threshold)
            speech_ratio = sum(self.speech_buffer) / len(self.speech_buffer)

            return speech_ratio > 0.7  # 70% последних чанков должны содержать речь
        except Exception as e:
            print(f"Ошибка VAD: {str(e)}")
            return False

    def reset(self):
        """Сброс состояния VAD"""
        self.speech_buffer.clear()