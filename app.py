import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import json
from tensorflow.keras.layers import TextVectorization
import os
import gdown
import requests

# Page configuration
st.set_page_config(page_title="Toxicity Detector", layout="wide")
st.title("Comment Toxicity Detector")

# Load configuration
@st.cache_resource
def load_config():
    try:
        with open("model_config.json", "r") as f:
            return json.load(f)
    except:
        return {
            "max_words": 50000,
            "max_len": 300,
            "label_names": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        }

config = load_config()
MAX_WORDS = config["max_words"]
MAX_LEN = config["max_len"]
LABELS = config["label_names"]

# Load model from Google Drive
@st.cache_resource
def load_model():
    model_path = "best_toxicity_model.keras"
    file_id = "1_MYD80RuzpVyr0XbevB6e6G8YwVFaoiA"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive...")
        try:
            gdown.download(download_url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Model download failed: {str(e)}")
            return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Could not load model: {str(e)}")
        return None

# Load vectorizer vocabulary from GitHub
@st.cache_resource
def load_vectorizer():
    vocab_url = "https://raw.githubusercontent.com/ahmedsh711/toxicity-detector-app/main/vectorizer_vocab.pkl"
    vocab_path = "vectorizer_vocab.pkl"

    if not os.path.exists(vocab_path):
        st.info("Downloading vectorizer vocabulary from GitHub...")
        try:
            r = requests.get(vocab_url)
            with open(vocab_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.error(f"Vectorizer download failed: {str(e)}")
            return None

    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        vec = TextVectorization(
            max_tokens=MAX_WORDS,
            output_sequence_length=MAX_LEN,
            output_mode="int",
            standardize="lower_and_strip_punctuation"
        )
        vec.set_vocabulary(vocab)
        st.success("Vectorizer loaded successfully.")
        return vec
    except Exception as e:
        st.error(f"Could not load vectorizer: {str(e)}")
        return None

# Initialize model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

if not model or not vectorizer:
    st.stop()

# Sidebar controls
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.65, 0.05)
st.sidebar.write("Adjust threshold to make detection stricter or more lenient.")

# Analysis functions
def analyze(text):
    vec = vectorizer([text])
    preds = model.predict(vec, verbose=0)[0]
    preds = (preds - 0.3) / 0.7  # calibration
    preds = preds.clip(0, 1)
    return preds

def summarize(text, preds):
    is_toxic = any(preds > threshold)
    toxic_labels = [LABELS[i] for i, p in enumerate(preds) if p > threshold]
    return {
        "Comment": text[:80] + "..." if len(text) > 80 else text,
        "Status": "Toxic" if is_toxic else "Clean",
        "Confidence": f"{max(preds):.1%}",
        "Categories": ", ".join(toxic_labels) if toxic_labels else "None"
    }

# Tabs for analysis
tab1, tab2 = st.tabs(["Single Comment", "Multiple Comments"])

# Single comment analysis
with tab1:
    txt = st.text_area("Enter a comment:")
    if st.button("Analyze"):
        if not txt.strip():
            st.warning("Please enter a comment first.")
        else:
            preds = analyze(txt)
            st.write("---")
            if any(preds > threshold):
                st.error("Toxic content detected.")
            else:
                st.success("Clean comment.")
            st.subheader("Prediction Details:")
            for label, p in zip(LABELS, preds):
                st.write(f"{label.title()}: {p:.1%}")
                st.progress(float(p))

# Multiple comment analysis
with tab2:
    mode = st.radio("Input method:", ["Paste", "Upload CSV"])

    if mode == "Paste":
        data = st.text_area("Paste comments (one per line):")
        if st.button("Analyze All"):
            if not data.strip():
                st.warning("Please paste at least one comment.")
            else:
                comments = [c.strip() for c in data.split("\n") if c.strip()]
                results = [summarize(c, analyze(c)) for c in comments]
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False), "results.csv")

    else:
        file = st.file_uploader("Upload CSV with a column named 'comment_text'", type=["csv"])
        if st.button("Analyze File") and file:
            df = pd.read_csv(file)
            if "comment_text" not in df.columns:
                st.error("CSV must contain a column named 'comment_text'.")
            else:
                results = [summarize(c, analyze(c)) for c in df["comment_text"].astype(str)]
                out = pd.DataFrame(results)
                st.dataframe(out, use_container_width=True)
                st.download_button("Download CSV", out.to_csv(index=False), "results.csv")
