import mlflow
import pandas as pd

# Load Production model from MLflow Registry
model = mlflow.sklearn.load_model("models:/ecommerce_best_model/Production")

# Example input (same features you used for training)
sample = pd.DataFrame({
    "Avg. Session Length": [34.5],
    "Time on App": [12.3],
    "Time on Website": [15.1],
    "Length of Membership": [5.2]
})

# Predict
prediction = model.predict(sample)
print(f"Predicted Yearly Amount Spent: ${prediction[0]:,.2f}")
