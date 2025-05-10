
# 🚀 E-commerce ML Model Deployment Pipeline

This project demonstrates an end-to-end Machine Learning deployment workflow using:

- ✅ Python + MLflow for training and experiment tracking  
- ✅ Docker for containerization  
- ✅ GitHub Actions for CI/CD automation  
- ✅ AWS EC2 for production deployment

---

## 📁 Project Structure

```
.
├── train.py                  # Train and log models to MLflow
├── app.py / main.py         # Optional: API for serving predictions
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── mlruns/                  # Local MLflow tracking directory (optional)
├── .github/
│   └── workflows/
│       └── build-train-deploy.yml  # GitHub Actions workflow
└── README.md                # This file
```

---

## 🧠 What This Project Does

1. **Trains a machine learning model** on local data
2. **Logs the model to MLflow**, including metrics and parameters
3. **Builds a Docker image** to serve the model (via FastAPI/Flask or as a CLI)
4. **Pushes the Docker image** to Docker Hub
5. **SSHs into an AWS EC2 server** and runs the Docker container
6. **Automates the entire process** through GitHub Actions (CI/CD)

---

## 🔄 End-to-End Flow (CI/CD Breakdown)

### Trigger
Pushing code to the `main` or `train` branch automatically starts the pipeline.

### GitHub Actions Workflow Steps

| Step | Description |
|------|-------------|
| ✅ Set up job | Initializes the GitHub runner |
| 📥 Checkout code | Clones your repo to the runner |
| 🐍 Set up Python | Installs Python version and environment |
| 📦 Install dependencies | Runs `pip install -r requirements.txt` |
| 🏋️‍♂️ Run training and log to MLflow | Executes `train.py` and logs metrics to MLflow |
| 🔐 Login to Docker Hub | Uses GitHub Secrets to authenticate |
| 📦 Build & Push Docker image | Builds container and pushes to Docker Hub |
| 🧳 SSH to EC2 and Deploy | Connects to EC2, pulls image, and runs container |
| 🧹 Post-cleanup | Frees up resources |

---

## 📌 Key Files Explained

### `train.py`
Trains a model and logs it using MLflow.
```python
import mlflow
mlflow.log_metric("mae", mae)
mlflow.sklearn.log_model(model, "model")
```

### `Dockerfile`
Defines the environment for the container.
```dockerfile
FROM python:3.10
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### `.github/workflows/build-train-deploy.yml`
GitHub Actions CI/CD automation file.
```yaml
steps:
  - name: Checkout code
    uses: actions/checkout@v3

  - name: Set up Python
    uses: actions/setup-python@v4

  - name: Install dependencies
    run: pip install -r requirements.txt

  - name: Train and log model
    run: python train.py

  - name: Login to Docker Hub
    run: echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin

  - name: Build and Push Docker Image
    run: |
      docker build -t ${{ secrets.DOCKER_USERNAME }}/ecommerce-model:latest .
      docker push ${{ secrets.DOCKER_USERNAME }}/ecommerce-model:latest

  - name: SSH and Deploy on EC2
    run: |
      ssh -o StrictHostKeyChecking=no -i key.pem ec2-user@${{ secrets.EC2_HOST }} "
        docker pull ${{ secrets.DOCKER_USERNAME }}/ecommerce-model:latest &&
        docker stop ecommerce || true &&
        docker rm ecommerce || true &&
        docker run -d --name ecommerce -p 80:80 ${{ secrets.DOCKER_USERNAME }}/ecommerce-model:latest
      "
```

---

## 🔐 Required GitHub Secrets

| Name | Description |
|------|-------------|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password or token |
| `EC2_HOST` | Public IP or DNS of your EC2 instance |
| `EC2_SSH_KEY` | SSH private key to connect to EC2 |
| `EC2_USERNAME` | (usually `ec2-user` for Amazon Linux) |

---

## 🌍 EC2 Setup (Manual, Once)

1. Launch an EC2 instance (Amazon Linux)
2. Install Docker:
```bash
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
```
3. Add your public SSH key
4. Make port `80` accessible in the security group

---

## ✅ How to Use

1. Clone the repo
2. Set up GitHub Secrets
3. Push code to `main`
4. GitHub Actions will auto-train, build, and deploy

---

## 🧪 How to Test the Deployment

After deployment, you can test if the model/API is live by:

### Option 1: Web Browser (If FastAPI/Flask is used)
```bash
http://<EC2_PUBLIC_IP>
```
This should show a welcome page or API UI (e.g., Swagger UI if FastAPI).

### Option 2: Curl
```bash
curl http://<EC2_PUBLIC_IP>/predict -X POST -H "Content-Type: application/json" -d '{"feature_1": 5, "feature_2": 2}'
```

### Option 3: Python Request
```python
import requests
response = requests.post("http://<EC2_PUBLIC_IP>/predict", json={"feature_1": 5, "feature_2": 2})
print(response.json())
```

---

## 💬 What to Say in an Interview

> “This project automates the full ML lifecycle — from training to cloud deployment. I built a GitHub Actions CI/CD pipeline that triggers model training, logs experiments with MLflow, containerizes the app with Docker, and deploys it to AWS EC2. It’s designed for real-world scalability and reproducibility.”

---

## 🧠 Future Improvements

- Add FastAPI prediction endpoint `/predict`
- Move MLflow to remote PostgreSQL + S3
- Integrate Streamlit dashboard
- Replace EC2 with ECS or Lambda for scalability
