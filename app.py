import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import json
from tensorflow.keras.layers import TextVectorization
import os
import gdown

# Page config
st.set_page_config(page_title="Toxicity Detector", layout="wide")
st.title("Comment Toxicity Detector")

# Load config
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

# Load model
@st.cache_resource
def load_model():
    model_path = "best_toxicity_model.keras"
    file_id = "1YuR2RRT0l9rCEX3ahsuicpREnjzM24X6"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive...")
        try:
            gdown.download(download_url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Could not load model: {str(e)}")
        return None

# Load vectorizer
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
        st.success("Vectorizer loaded successfully.")
        return vec
    except Exception as e:
        st.error(f"Could not load vectorizer: {str(e)}")
        return None

# Initialize
model = load_model()
vectorizer = load_vectorizer()

if not model or not vectorizer:
    st.stop()

# Sidebar
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

# Tabs
tab1, tab2 = st.tabs(["Single Comment", "Multiple Comments"])

# Single comment
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

# Multiple comments
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
        elif st.button("Analyze All") and not data.strip():
            st.warning("Please paste at least one comment.")

    else:
        file = st.file_uploader("Upload CSV with a column named 'comment_text'", type=["csv"])
        if file and st.button("Analyze File"):
            df = pd.read_csv(file)
            if "comment_text" not in df.columns:
                st.error("CSV must contain a column named 'comment_text'.")
            else:
                df["Results"] = df["comment_text"].apply(lambda x: summarize(x, analyze(x)))
                results = pd.json_normalize(df["Results"])
                st.dataframe(results, use_container_width=True)
                st.download_button("Download CSV", results.to_csv(index=False), "results.csv")
