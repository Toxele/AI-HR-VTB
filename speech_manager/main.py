import pyaudio
import numpy as np
import time
import io
import wave
import torch
#torch.hub.list('snakers4/silero-vad', force_reload=True) # раскоментить, если будет ошибка загрузки или типа того
import torch
import torchaudio
print(f"PyTorch version: {torch.__version__}")
print(f"TorchAudio version: {torchaudio.__version__}")
print("Все модули загружены успешно!")
from collections import deque
from stt_service import WhisperSTT
from tts_service import VoskTTS
from vad_service import SileroVAD
from dialog_service import SmartTurn
from context_manager import ContextManager
#from silero_vad import SileroVAD

vad = SileroVAD(threshold=0.015)
# Инициализация сервисов
stt = WhisperSTT()
tts = VoskTTS()
vad = SileroVAD(threshold=0.015)
endpoint_detector = SmartTurn()

# Параметры аудио
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000  # Для микрофона и STT
OUTPUT_RATE = 22050  # Для TTS воспроизведения
CHUNK = 512

# Параметры буфера предварительной записи
PRE_RECORD_SECONDS = 3  # Сохранять последние 3 секунды аудио
PRE_RECORD_SIZE = int(PRE_RECORD_SECONDS * INPUT_RATE / CHUNK)  # Размер буфера в чанках

# Состояние системы
pre_record_buffer = deque(maxlen=PRE_RECORD_SIZE)  # Кольцевой буфер для предварительной записи
audio_buffer = np.array([], dtype=np.float32)
is_recording = False
silence_chunks = 0
SILENCE_THRESHOLD = 15  # Количество тихих чанков для остановки записи


def play_audio(audio_bytes):
    """Воспроизведение аудио через PyAudio с частотой 22050 Гц"""
    p = pyaudio.PyAudio()

    # Открываем поток с частотой 22050 Гц для TTS аудио
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=OUTPUT_RATE,
                    output=True)

    # Читаем WAV файл из байтов
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        # Проверяем, что частота дискретизации соответствует ожидаемой
        if wf.getframerate() != OUTPUT_RATE:
            print(f"Предупреждение: частота TTS аудио {wf.getframerate()} Гц не совпадает с ожидаемой {OUTPUT_RATE} Гц")

        # Читаем и воспроизводим все кадры
        data = wf.readframes(wf.getnframes())
        stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


def process_audio(audio_data):
    """Обработка аудиоданных: STT -> TTS -> Воспроизведение"""
    try:
        # STT
        text = stt.transcribe_audio_buffer(
            audio_data.astype(np.float32),
            sample_rate=INPUT_RATE,
            language="ru"
        )
        print(f"Распознано: {text}")

        # TTS
        audio_bytes = tts.synthesize_to_buffer(text, speaker_id=4)

        # Воспроизведение
        play_audio(audio_bytes)

    except Exception as e:
        print(f"Ошибка обработки аудио: {str(e)}")


# Инициализация аудиопотока для микрофона
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=INPUT_RATE,
                input=True,
                frames_per_buffer=CHUNK)


resume_filename = "Образец резюме 1 Бизнес аналитик.rtf"
vacancy_filename = "Описание бизнес аналитик.docx"
context_manager = ContextManager(resume_filename=resume_filename, vacancy_filename=vacancy_filename)

print("Система готова к работе. Говорите...")
try:
    while True:
        # Чтение аудиоданных с микрофона
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Всегда добавляем чанк в буфер предварительной записи
        pre_record_buffer.append(audio_chunk)

        # Проверка на наличие речи
        if vad.is_speech(audio_chunk):
            if not is_recording:
                print("Обнаружена речь, начинаю запись...")

                # Добавляем буфер предварительной записи к началу
                if pre_record_buffer:
                    audio_buffer = np.concatenate(list(pre_record_buffer))
                else:
                    audio_buffer = audio_chunk

                is_recording = True
                silence_chunks = 0
            else:
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                silence_chunks = 0
        else:
            if is_recording:
                silence_chunks += 1
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                # Проверка endpointing
                endpoint_result = endpoint_detector.predict_endpoint(audio_buffer)
                if endpoint_result["prediction"] == 1 or silence_chunks > SILENCE_THRESHOLD:
                    print("Речь завершена, обрабатываю аудио...")
                    process_audio(audio_buffer)
                    # Сброс состояния
                    is_recording = False
                    audio_buffer = np.array([], dtype=np.float32)
                    vad.reset()
                    print("Готов к следующему запросу...")

except KeyboardInterrupt:
    print("\nЗавершение работы...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()