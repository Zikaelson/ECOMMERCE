import mlflow

# Replace with your actual model name and version number
model_name = "ecommerce_best_model"
model_version = 2  # <- Replace with the version you want to promote

client = mlflow.tracking.MlflowClient()

# Promote to "Production"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production"
)

print(f"✅ Model {model_name} version {model_version} promoted to Production.")
