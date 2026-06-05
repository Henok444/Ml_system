# Crop Recommendation API

## Overview

This project deploys a machine learning model for crop recommendation using FastAPI.

The model predicts the most suitable crop based on soil nutrients and environmental conditions.

Features used:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

## Model

The API uses the best-performing ensemble model from the Precision Agriculture project.

Algorithms evaluated:

* Random Forest
* XGBoost

Evaluation metric:

* F1 Score

Best Results:

* Random Forest: 0.9932 F1 Score
* XGBoost: 0.9885 F1 Score

## Installation

Install dependencies:

pip install -r requirements.txt

## Run the API

uvicorn app:app --reload

## API Documentation

After starting the server, open:

http://127.0.0.1:8000/docs

FastAPI automatically generates interactive documentation.

## Example Request

POST /predict

Parameters:

N=90

P=42

K=43

temperature=20.8

humidity=82.0

ph=6.5

rainfall=202.9

## Example Response

{
"recommended_crop": "rice"
}

## Project Workflow

1. Data preprocessing
2. Exploratory Data Analysis
3. Hyperparameter tuning
4. Ensemble model training
5. Model evaluation
6. Feature importance analysis
7. API deployment using FastAPI
