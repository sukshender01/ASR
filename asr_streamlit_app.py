import streamlit as st
import sounddevice as sd
import numpy as np
import json
from faster_whisper import WhisperModel
from queue import Queue
import threading
import matplotlib.pyplot as plt

LANGUAGE = st.sidebar.selectbox("Language", options=["en", "ja"], index=0)
MODEL_SIZE = "base"
CHUNK_DURATION = 0.7
SAMPLE_RATE = 16000
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_DURATION)
DEVICE = "cpu"

st.set_page_config(page_title="Real-Time ASR with Waveform", layout="wide")
st.title("🎙️ Real-Time ASR (EN/JA) + Audio Waveform + JSON Export")

start_button = st.button("▶️ Start Listening")
stop_button = st.button("⏹️ Stop Listening")

status_box = st.empty()
output_text = st.empty()
audio_plot = st.empty()
volume_bar = st.empty()

@st.cache_resource
def load_model():
    model = WhisperModel(MODEL_SIZE, compute_type="int8", device=DEVICE)
    return model

model = load_model()
status_box.success("✅ Model loaded (CPU only)")

audio_queue = Queue()
text_buffer = []
json_buffer = []

def asr_loop():
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break

        mono = np.mean(audio_chunk, axis=1) if audio_chunk.ndim > 1 else audio_chunk

        segments, _ = model.transcribe(
            mono,
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True
        )

        for segment in segments:
            text = segment.text.strip()
            words = [{
                "word": w.word,
                "start": round(w.start, 2),
                "end": round(w.end, 2)
            } for w in segment.words] if segment.words else []

            if text:
                output_text.markdown(f"**📝 Transcribed:** {text}")
                text_buffer.append(text)
                json_buffer.append({
                    "text": text,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "words": words
                })

def update_waveform(chunk):
    mono = np.mean(chunk, axis=1) if chunk.ndim > 1 else chunk

    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.plot(mono, linewidth=0.8)
    ax.set_ylim([-1.0, 1.0])
    ax.set_axis_off()
    audio_plot.pyplot(fig)

    volume = np.sqrt(np.mean(mono ** 2))
    volume_bar.progress(min(volume * 5, 1.0), text=f"🎚️ Volume: {volume:.2f}")

def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ {status}")
    audio_queue.put(indata.copy())
    update_waveform(indata)

asr_thread = None
stream = None

if start_button:
    text_buffer.clear()
    json_buffer.clear()
    asr_thread = threading.Thread(target=asr_loop, daemon=True)
    asr_thread.start()
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE)
    stream.start()
    status_box.info("🎧 Listening...")

if stop_button:
    if stream:
        stream.stop()
        stream.close()
    audio_queue.put(None)
    if asr_thread:
        asr_thread.join()
    status_box.warning("🛑 Stopped.")
    st.download_button("⬇️ Download Transcript (.txt)", data="\n".join(text_buffer), file_name="transcript.txt")
    json_str = json.dumps(json_buffer, ensure_ascii=False, indent=2)
    st.download_button("⬇️ Download Full Transcript (.json)", data=json_str, file_name="transcript.json")