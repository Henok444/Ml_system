from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(
    title="Crop Recommendation API",
    description="Predict the most suitable crop using soil and environmental features.",
    version="1.0"
)

model = joblib.load("crop_model.pkl")


@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running"}


@app.post("/predict")
def predict(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
):

    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = model.predict(input_data)

    return {
        "recommended_crop": prediction[0]
    }