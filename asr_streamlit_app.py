import os
import streamlit as st
import torch
import whisper
from datasets import load_dataset, Audio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

# ---- CONFIG ----
DATASET_NAME = "mozilla-foundation/common_voice_12_0"
LANG = "ja"
SPLIT = "train"
SAMPLERATE = 16000

@st.cache_resource
def load_models():
    asr = whisper.load_model("small")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return asr, tokenizer, model, tts

asr_model, translation_tokenizer, translation_model, tts = load_models()

def compute_bleu(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    matches = sum(1 for word in hyp_words if word in ref_words)
    bleu = matches / len(hyp_words) if hyp_words else 0.0
    return bleu

def estimate_token_delays(ref_tokens, hyp_tokens):
    delays = []
    ref = ref_tokens.tolist()
    hyp = hyp_tokens.tolist()
    for i, h in enumerate(hyp):
        try:
            delay = abs(i - ref.index(h))
        except ValueError:
            delay = len(ref)
        delays.append(delay)
    avg_delay = sum(delays) / len(delays) if delays else 0
    accuracy = sum(1 for h in hyp if h in ref) / len(hyp) if hyp else 0
    return delays, avg_delay, accuracy

def translate_text(text, tokenizer, model):
    src_lang = "ja_XX"
    tgt_lang = "en_XX"
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=512
    )
    translated = tokenizer.decode(generated[0], skip_special_tokens=True)
    ref_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    hyp_tokens = generated[0]
    token_delay, avg_token_delay, token_accuracy = estimate_token_delays(ref_tokens, hyp_tokens)
    bleu_score = compute_bleu(text, translated)
    metrics = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "bleu": bleu_score,
        "accuracy": token_accuracy,
        "avg_delay": avg_token_delay,
        "token_delay": token_delay
    }
    return translated, metrics

def main():
    st.set_page_config(page_title="Streaming Speech Translator + Metrics", layout="centered")
    st.title("🇯🇵📡 Japanese Speech Translator with WAV Output & Metrics")
    st.caption("Transcribes, translates, synthesizes and evaluates in real time using Common Voice (JA).")

    num_samples = st.number_input("Number of samples to process", min_value=1, max_value=10, value=3)
    output_dir = st.text_input("Output folder", value="translation_tts_metrics_results")
    os.makedirs(output_dir, exist_ok=True)

    if st.button("🚀 Start Streaming Translation"):
        dataset = load_dataset(DATASET_NAME, LANG, split=SPLIT, streaming=True)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLERATE))
        my_bar = st.progress(0, text="Starting translation and audio synthesis...")
        for idx, sample in enumerate(dataset):
            if idx >= num_samples:
                break

            audio_arr = sample["audio"]["array"]
            result = asr_model.transcribe(audio_arr, language="ja")
            transcription = result["text"]

            translated, metrics = translate_text(transcription, translation_tokenizer, translation_model)

            wav_path = os.path.join(output_dir, f"sample_{idx+1}_translated.wav")
            tts.tts_to_file(text=translated, file_path=wav_path, language="en")

            text_path = os.path.join(output_dir, f"sample_{idx+1}_results.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription: {transcription}\n")
                f.write(f"Translation: {translated}\n")
                f.write(f"TTS WAV File: {wav_path}\n")
                f.write("Metrics:\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")

            percent = int((idx + 1) * 100 / num_samples)
            my_bar.progress(percent, text=f"{idx+1}/{num_samples} processed")

            st.markdown(f"### Sample {idx+1}")
            st.write(f"**Transcription:** {transcription}")
            st.write(f"**Translation:** {translated}")
            st.audio(wav_path, format="audio/wav", start_time=0)

            m = metrics
            st.subheader("📊 Evaluation Metrics")
            st.markdown(f"- **Source Language**: `{m['src_lang']}`")
            st.markdown(f"- **Target Language**: `{m['tgt_lang']}`")
            st.markdown(f"- **Average Token Delay**: `{m['avg_delay']:.2f}`")
            st.markdown(f"- **Token Accuracy**: `{m['accuracy'] * 100:.2f}%`")
            st.markdown(f"- **BLEU Score**: `{m['bleu'] * 100:.2f}%`")
            st.markdown(f"- **Token Delay Values**: `{m['token_delay']}`")
            st.divider()

        my_bar.empty()
        st.success(f"Done! All translations and metrics saved in `{output_dir}`.")

if __name__ == "__main__":
    main()
