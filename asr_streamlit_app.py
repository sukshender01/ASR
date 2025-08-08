import streamlit as st
import os
import tempfile
import zipfile
import requests
import torch
import torchaudio
import shutil
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

# Initialize session state logs
if 'logs' not in st.session_state:
    st.session_state.logs = []

def log(msg):
    st.session_state.logs.append(f"📝 {msg}")
    with st.expander("Logs", expanded=True):
        for l in st.session_state.logs[-10:]:
            st.markdown(l)

# Default Dataset URL (ZIP of audio files + transcript)
DEFAULT_DATASET_URL = "https://huggingface.co/datasets/openslr/slr22/resolve/main/test-clean.zip"  # Replace with your actual default dataset zip

# Load Whisper Model
@st.cache_resource
def load_asr_model():
    log("🔄 Loading Whisper ASR model (small)...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    return model, processor

# Load Translator
@st.cache_resource
def load_translation_model():
    log("🔄 Loading translation model (Japanese-English)...")
    model_name = "Helsinki-NLP/opus-mt-ja-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def download_and_extract(url):
    log(f"📥 Downloading dataset from {url}...")
    response = requests.get(url)
    tempdir = tempfile.mkdtemp()
    zip_path = os.path.join(tempdir, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)
    
    log("🗜️ Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tempdir)

    audio_files = []
    for root, dirs, files in os.walk(tempdir):
        for file in files:
            if file.endswith(".flac") or file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))

    return audio_files

def transcribe(audio_path, model, processor):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def translate_text(text, model, tokenizer):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def compute_metrics(reference, hypothesis):
    smooth_fn = SmoothingFunction().method1
    bleu = sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smooth_fn)
    accuracy = sum(a == b for a, b in zip(reference.split(), hypothesis.split())) / max(1, len(reference.split()))
    lagging = abs(len(reference.split()) - len(hypothesis.split()))
    avg_token_delay = lagging / max(1, len(hypothesis.split()))
    return bleu, accuracy, lagging, avg_token_delay

# UI
st.title("🎧 Whisper-based Speech Translation (JP/EN)")
url_input = st.text_input("📎 Enter custom dataset ZIP URL (or leave blank for default):", "")

dataset_url = url_input.strip() if url_input else DEFAULT_DATASET_URL
if st.button("🚀 Start Processing"):
    try:
        asr_model, asr_processor = load_asr_model()
        nmt_model, nmt_tokenizer = load_translation_model()

        files = download_and_extract(dataset_url)
        st.success(f"Found {len(files)} audio files.")

        results = []

        for i, audio_file in enumerate(files[:10]):  # Limit to 10 files for demo
            st.markdown(f"### 🔊 File {i+1}: `{os.path.basename(audio_file)}`")
            start = time.time()
            transcription = transcribe(audio_file, asr_model, asr_processor)
            lang = "ja" if any(ord(c) > 128 for c in transcription) else "en"
            translation = translate_text(transcription, nmt_model, nmt_tokenizer)
            bleu, acc, lag, delay = compute_metrics(transcription, translation)
            duration = time.time() - start

            st.markdown(f"**🎤 Transcription:** `{transcription}`")
            st.markdown(f"**🌐 Translation:** `{translation}`")
            st.markdown(f"**📊 Metrics:**")
            st.markdown(f"- Detected Language: `{lang}`")
            st.markdown(f"- BLEU Score: `{bleu:.4f}`")
            st.markdown(f"- Accuracy: `{acc:.2f}`")
            st.markdown(f"- Token Lagging: `{lag}`")
            st.markdown(f"- Average Token Delay: `{delay:.2f}`")
            st.markdown(f"- Processing Time: `{duration:.2f} sec`")

            results.append((audio_file, transcription, translation, bleu, acc, lag, delay))

    except Exception as e:
        st.error(f"❌ Failed to process dataset: {e}")
