from vosk_tts import Model, Synth
import io
import wave


class VoskTTS:
    def __init__(self, model_name: str = "vosk-model-tts-ru-0.9-multi"):
        self.model = Model(model_name=model_name)
        self.synth = Synth(self.model)

    def synthesize_to_file(self, text: str, output_path: str, speaker_id: int = 4) -> None:
        """
        Синтезирует речь и сохраняет в файл
        Args:
            text: текст для синтеза
            output_path: путь для сохранения аудиофайла
            speaker_id: ID голоса (0-4)
        """
        try:
            self.synth.synth(text, speaker_id=speaker_id, oname=output_path)
        except Exception as e:
            raise Exception(f"Ошибка синтеза речи: {str(e)}")

    def synthesize_to_buffer(self, text: str, speaker_id: int = 4) -> bytes:
        """
        Синтезирует речь и возвращает в виде байтов
        Args:
            text: текст для синтеза
            speaker_id: ID голоса (0-4)
        Returns:
            аудиоданные в формате WAV
        """
        try:
            # Создаем временный буфер в памяти
            with io.BytesIO() as wav_buffer:
                # Синтезируем речь напрямую в буфер
                audio_array = self.synth.synth_audio(text, speaker_id=speaker_id)

                # Записываем WAV-заголовок и данные в буфер
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit audio
                    wav_file.setframerate(22050)
                    wav_file.writeframes(audio_array.tobytes())

                # Возвращаем содержимое буфера как байты
                return wav_buffer.getvalue()

        except Exception as e:
            raise Exception(f"Ошибка синтеза речи: {str(e)}")