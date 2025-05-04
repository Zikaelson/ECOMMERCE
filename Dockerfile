# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code only (no mlruns/)
COPY streamlit_app/ ./streamlit_app/

# Copy .env file if needed (or inject via GitHub secrets)
# COPY .env .env

# Use remote MLflow server from .env (or from secrets)
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
