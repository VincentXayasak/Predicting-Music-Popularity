import gradio as gr
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("lgbm_pipeline.pkl")

# Predict function
def predict_popularity(file):
    df = pd.read_csv(file.name)
    predictions = pipeline.predict(df).tolist()
    return {"predictions": predictions}

# Gradio interface
app = gr.Interface(
    fn=predict_popularity,
    inputs=gr.File(label="Upload CSV with Song Features"),
    outputs=gr.JSON(label="Predicted Popularity"),
    title="Spotify Song Popularity Predictor",
    description="Upload a CSV with the appropriate features to predict song popularity using a LightGBM model."
)

app.launch()