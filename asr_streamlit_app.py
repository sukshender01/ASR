# asr_streamlit_app.py
import streamlit as st
import torchaudio
import torch
import numpy as np
import soundfile as sf
import tempfile
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import sacrebleu

st.title("Real-Time ASR + Translation with HuggingFace")
st.markdown("This demo uses Whisper + Seq2Seq NMT pipeline to translate Japanese <-> English.")

# Load processor and model
asr_processor = AutoProcessor.from_pretrained("openai/whisper-small")
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
asr_model.eval()

# Load NMT model
nmt_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
nmt_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
nmt_model.eval()

def transcribe(audio):
    inputs = asr_processor(audio[0], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = asr_model.generate(inputs.input_features)
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def translate(text):
    inputs = nmt_tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        generated_tokens = nmt_model.generate(**inputs)
    return nmt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

option = st.radio("Choose Dataset Source", ("Use default HuggingFace dataset", "Upload your own dataset URL"))

if option == "Use default HuggingFace dataset":
    dataset = load_dataset("gsarti/fleurs", "ja_jp", split="test[:1]")
    audio = dataset[0]['audio']['array'], dataset[0]['audio']['sampling_rate']
    ref_text = dataset[0]['text']
    st.audio(dataset[0]['audio']['path'], format='audio/wav')
else:
    url = st.text_input("Enter the HuggingFace dataset audio file URL:")
    if url:
        audio, _ = torchaudio.load(url)
        audio = audio.mean(dim=0).numpy(), 16000  # convert to mono
        st.audio(url, format='audio/wav')
        ref_text = st.text_input("Enter the reference transcription for BLEU calculation:")

if 'audio' in locals():
    st.write("### Transcription:")
    transcription = transcribe(audio)
    st.success(transcription)

    st.write("### Translation:")
    translation = translate(transcription)
    st.success(translation)

    if ref_text:
        st.write("### BLEU Score:")
        bleu = sacrebleu.corpus_bleu([translation], [[ref_text]])
        st.metric(label="BLEU Score", value=f"{bleu.score:.2f}")
