# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code and model registry
COPY streamlit_app/ ./streamlit_app/
COPY mlruns/ ./mlruns/

# Set MLflow tracking path (for local registry use)
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
