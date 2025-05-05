import os
import sys
import shutil
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns  
import warnings
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ✅ Load environment variables from .env file
load_dotenv()

# 👇 Allow imports from utils.py
sys.path.append(os.path.abspath("../src"))
from utils import load_data

# ✅ Set tracking URI for backend store (use SQLite for model registry)
mlflow.set_tracking_uri('sqlite:///mlruns.db')  # Use local SQLite for model registry
mlflow.set_experiment("ecommerce-experiment")  # This maps to experiment ID 0

# ✅ Set artifact location (S3) for model artifacts
mlflow.set_artifact_uri('s3://ecommerce-mlflow-artifacts')  # Artifact location in S3

# ✅ Clean any old registry to avoid Windows-path issues
registry_path = "./mlruns/models/ecommerce_best_model"
if os.path.exists(registry_path):
    shutil.rmtree(registry_path)

# ✅ Load dataset
df = load_data("data/customerdata.csv")

# ✅ Feature and target selection
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# ---------------------- Linear Regression ---------------------- #
with mlflow.start_run(run_name="Linear Regression") as run:
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2", r2_score(y_test, preds))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))

    mlflow.sklearn.log_model(model, "model", input_example=pd.DataFrame(X_train.iloc[0]).T)
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "ecommerce_best_model")

    # Promote latest version to Production
    client = MlflowClient()
    version = client.get_latest_versions("ecommerce_best_model", stages=["None"])[0].version
    client.transition_model_version_stage(
        name="ecommerce_best_model",
        version=version,
        stage="Production"
    )
    print(f"Model version {version} promoted to Production.")

# ---------------------- Random Forest ---------------------- #
with mlflow.start_run(run_name="Random Forest"):
    model = RandomForestRegressor(n_estimators=100, random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("r2", r2_score(y_test, preds))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))
    mlflow.sklearn.log_model(model, "model", input_example=pd.DataFrame(X_train.iloc[0]).T)

# ---------------------- Gradient Boosting ---------------------- #
with mlflow.start_run(run_name="Gradient Boosting"):
    model = GradientBoostingRegressor(random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_metric("r2", r2_score(y_test, preds))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))
    mlflow.sklearn.log_model(model, "model", input_example=pd.DataFrame(X_train.iloc[0]).T)
