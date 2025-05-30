import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wavfile
import os

DURATION = 5
SAMPLERATE = 16000

# 녹음
print("🎤 말해주세요 (5초)...")
audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='int16')
sd.wait()
print("✅ 녹음 완료")

# 저장
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "recorded.wav")
wavfile.write(file_path, SAMPLERATE, audio)

# 인식
model = whisper.load_model("base")
result = model.transcribe(file_path, language="ko")
print("📝 인식 결과:", result["text"])
