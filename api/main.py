# api/main.py

import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Define expected input format using Pydantic
class InputData(BaseModel):
    Avg_Session_Length: float
    Time_on_App: float
    Time_on_Website: float
    Length_of_Membership: float

# Initialize FastAPI app
app = FastAPI()

# Load the production model from MLflow Registry
model = mlflow.sklearn.load_model("models:/ecommerce_best_model/Production")

@app.get("/")
def root():
    return {"message": "Ecommerce model is live 🚀"}

@app.post("/predict")
def predict(input: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([{
        "Avg. Session Length": input.Avg_Session_Length,
        "Time on App": input.Time_on_App,
        "Time on Website": input.Time_on_Website,
        "Length of Membership": input.Length_of_Membership
    }])

    # Make prediction
    pred = model.predict(df)[0]
    return {"predicted_spend": round(pred, 2)}
