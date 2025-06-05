import gradio as gr
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("lgbm_pipeline.pkl")

def predict_popularity(file):
    df = pd.read_csv(file.name)
    preds = pipeline.predict(df).tolist()
    return preds

app = gr.Interface(
    fn=predict_popularity,
    inputs=gr.File(label="Upload CSV with Song Features"),
    outputs=gr.JSON(label="Predicted Popularity"),
    title="Spotify Song Popularity Predictor"
)

app.launch()