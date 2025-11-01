# Toxicity Detector App

Hey there! This is a simple Streamlit app built around the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) on Kaggle. The goal of that competition was to classify comments as toxic or not, based on categories like "toxic," "severe toxic," "obscene," "threat," "insult," or "identity hate." I trained a model on that dataset and wrapped it in this app to make it easy to check comments for toxicity.

## What Does It Do?
- **Single Comment Analysis**: Paste a comment, hit "Analyze," and get a breakdown of toxicity levels with progress bars.
- **Batch Analysis**: Paste multiple comments or upload a CSV file, and get a summary table of results. You can even download the output as CSV.
- **Custom Threshold**: Use the sidebar slider to set how sensitive the detection should be (e.g., 0.5 by default).

It's powered by a TensorFlow model and keeps things lightweight—no fancy bells and whistles, just straightforward toxicity detection.

## How to Set It Up Locally
1. Clone this repo: `git clone https://github.com/yourusername/toxicity-detector-app.git` (replace with your repo URL).
2. Install dependencies: Run `pip install -r requirements.txt` (needs Streamlit, TensorFlow, and Pandas).
3. Make sure you have these files in the folder:
   - `app.py` (the main script)
   - `best_toxicity_model.keras` (the trained model)
   - `vectorizer_vocab.pkl` (for text processing)
   - `model_config.json` (config settings, optional fallback in code)
4. Fire it up: `streamlit run app.py`

## Usage Tips
- For a single comment: Go to the "Single Comment" tab, type or paste, and analyze.
- For multiples: Use "Multiple Comments" tab—either paste lines or upload a CSV with a "comment_text" column.
- The app stops if it can't load the model or vectorizer, so double-check those files.

## Deployment
I deployed this to Streamlit Cloud for easy sharing. Just push to GitHub and connect it there—super quick. Check the live version [here](https://your-app-url.streamlit.app) (update with your actual URL).

## Limitations
- This is based on an older Kaggle dataset, so it might not catch everything in modern lingo.
- Model accuracy isn't perfect—use it as a tool, not gospel.
- If you're tweaking the model, retrain on fresh data for better results.

Feel free to fork and improve! If you spot issues, open a pull request. 😊
