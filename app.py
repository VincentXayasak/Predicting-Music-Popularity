import gradio as gr
import pandas as pd
import joblib
import os

# Path on Hugging Face will be in root dir
model_path = os.path.join(os.path.dirname(__file__), "lgbm_pipeline.pkl")
pipeline = joblib.load(model_path)

def predict_popularity(file):
    df = pd.read_csv(file.name)
    predictions = pipeline.predict(df).tolist()
    return {"predictions": predictions}

app = gr.Interface(
    fn=predict_popularity,
    inputs=gr.File(label="Upload CSV"),
    outputs=gr.JSON(label="Predicted Popularity"),
    title="Spotify Song Popularity Predictor"
)

app.launch()
