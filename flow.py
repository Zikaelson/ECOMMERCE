import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create dummy data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)

# Set remote tracking
mlflow.set_tracking_uri("http://13.58.154.74:5000")
mlflow.set_experiment("my-first-experiment")

# Log run
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2_score", model.score(X_test, y_test))
    
    # ✅ Log the model itself
    mlflow.sklearn.log_model(model, "model")
    print("Model and metrics logged!")
