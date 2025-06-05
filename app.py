import gradio as gr
import pandas as pd
import joblib
import os

# Load the model from the same directory
model_path = os.path.join(os.path.dirname(__file__), "lgbm_pipeline.pkl")
pipeline = joblib.load(model_path)

# Prediction function
def predict_popularity(file):
    df = pd.read_csv(file.name)
    predictions = pipeline.predict(df).tolist()
    return {"predictions": predictions}

# Gradio app
app = gr.Interface(
    fn=predict_popularity,
    inputs=gr.File(label="Upload CSV with Song Features"),
    outputs=gr.JSON(label="Predicted Popularity"),
    title="Spotify Song Popularity Predictor",
    description="Upload a CSV file with audio features to predict song popularity using a trained LightGBM model."
)

app.launch()