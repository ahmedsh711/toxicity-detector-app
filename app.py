import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import json
from tensorflow.keras.layers import TextVectorization
import os
import gdown

st.set_page_config(page_title="Toxicity Detector", layout="wide")
st.title("Comment Toxicity Detector")

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

@st.cache_resource
def load_model():
    model_path = "best_toxicity_model.keras"
    if not os.path.exists(model_path):
        st.info("Downloading model...")
        try:
            gdown.download("https://drive.google.com/file/d/1YuR2RRT0l9rCEX3ahsuicpREnjzM24X6/view?usp=drive_link", model_path, quiet=False)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except:
        st.error("Could not load model")
        return None

@st.cache_resource
def load_vectorizer():
    try:
        with open("vectorizer_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        vec = TextVectorization(
            max_tokens=MAX_WORDS,
            output_sequence_length=MAX_LEN,
            output_mode="int",
            standardize="lower_and_strip_punctuation"
        )
        vec.set_vocabulary(vocab)
        return vec
    except:
        st.error("Could not load vectorizer")
        return None

model = load_model()
vectorizer = load_vectorizer()

if not model or not vectorizer:
    st.stop()

threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.65, 0.05)

def analyze(text):
    vec = vectorizer([text])
    preds = model.predict(vec, verbose=0)[0]
    # Calibrate predictions
    preds = (preds - 0.3) / 0.7  # Shift and scale
    preds = preds.clip(0, 1)  # Keep in valid range
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

tab1, tab2 = st.tabs(["Single Comment", "Multiple Comments"])

with tab1:
    txt = st.text_area("Enter a comment:")
    if st.button("Analyze"):
        preds = analyze(txt)
        st.write("---")
        st.write("Toxic Content Detected" if any(preds > threshold) else "Clean Comment")
        for label, p in zip(LABELS, preds):
            st.write(f"{label.title()}: {p:.1%}")
            st.progress(float(p))

with tab2:
    mode = st.radio("Input method:", ["Paste", "Upload CSV"])
    if mode == "Paste":
        data = st.text_area("Paste comments (one per line):")
        if st.button("Analyze All") and data.strip():
            comments = [c.strip() for c in data.split("\n") if c.strip()]
            results = [summarize(c, analyze(c)) for c in comments]
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "results.csv")
    else:
        file = st.file_uploader("Upload CSV with 'comment_text' column", type=["csv"])
        if file and st.button("Analyze File"):
            df = pd.read_csv(file)
            if "comment_text" not in df.columns:
                st.error("Missing 'comment_text' column")
            else:
                results = [summarize(c, analyze(c)) for c in df["comment_text"].astype(str)]
                out = pd.DataFrame(results)
                st.dataframe(out, use_container_width=True)
                st.download_button("Download CSV", out.to_csv(index=False), "results.csv")
