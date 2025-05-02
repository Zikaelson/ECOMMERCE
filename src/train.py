import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# 👇 Allow imports from utils.py
sys.path.append(os.path.abspath("../src"))
from utils import load_data

# ✅ Load dataset
df = load_data("data/customerdata.csv")

# ✅ Select features and target
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# ✅ Enable autologging (optional but helpful)
mlflow.sklearn.autolog()

# ---------------------- Linear Regression ---------------------- #
with mlflow.start_run(run_name="Linear Regression"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Log metrics manually
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2", r2_score(y_test, preds))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))

    mlflow.sklearn.log_model(model, "model")

    # NEW: Register model
    mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id), "ecommerce_best_model")

# ---------------------- Random Forest ---------------------- #
with mlflow.start_run(run_name="Random Forest"):
    model = RandomForestRegressor(n_estimators=100, random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("r2", r2_score(y_test, preds))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))

    mlflow.sklearn.log_model(model, "model")


# ---------------------- Gradient Boosting ---------------------- #
with mlflow.start_run(run_name="Gradient Boosting"):
    model = GradientBoostingRegressor(random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_metric("r2", r2_score(y_test, preds))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))

    mlflow.sklearn.log_model(model, "model")
