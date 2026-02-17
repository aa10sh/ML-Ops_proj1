
# 🚗 Vehicle Insurance Prediction – End-to-End MLOps Project

An **end-to-end production-ready MLOps pipeline** that predicts whether a customer will purchase vehicle insurance.

This project demonstrates the **complete ML lifecycle**:

* Data ingestion from MongoDB ☁️
* Data validation & transformation ⚙️
* Model training & evaluation 🤖
* Model versioning on AWS S3 📦
* CI/CD with Docker + GitHub Actions 🚀
* Deployment using FastAPI on AWS EC2 🌐

---

🔗 1️⃣ Live Application Link (MOST IMPORTANT)

Once your EC2 app is running, add this at the top under the title:

## 🌍 Live Demo
🔗 http://54.88.188.192:5000

---

## 🐳 Docker Image (AWS ECR)

Image stored in AWS Elastic Container Registry:

286605287940.dkr.ecr.us-east-1.amazonaws.com/vehicleproj

## 📥 Dataset Stored in MongoDB Atlas
🔗 https://cloud.mongodb.com/v2/69784bc6e069530f8cc6cd1d#/explorer/69784c5bfe949047a2bf5ad8/Proj1/Proj1-data/find



# 🧠 Problem Statement

Insurance companies want to predict:

> **Will a customer buy vehicle insurance?**

This helps:

* Improve marketing strategy
* Reduce customer acquisition cost
* Increase conversion rate

This project builds a **scalable ML system** to solve this business problem.

---

📊 End-to-End MLOps Workflow
                        ┌────────────────────┐
                        │  Dataset (CSV)     │
                        └─────────┬──────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Upload to MongoDB Atlas │
                     └─────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Training Pipeline Trigger  │
                └─────────┬────────────────────┘
                          │
     ┌────────────────────┴────────────────────┐
     │                                         │
     ▼                                         ▼
┌──────────────┐                     ┌────────────────┐
│ Data Ingestion│                    │ Data Validation│
└──────┬────────┘                    └──────┬─────────┘
       ▼                                     ▼
┌──────────────┐                     ┌────────────────┐
│Data Transform │                    │ Model Training │
└──────┬────────┘                    └──────┬─────────┘
       ▼                                     ▼
┌──────────────────────────────────────────────┐
│        Model Evaluation vs Production Model  │
└──────────────┬───────────────────────────────┘
               ▼
        ┌──────────────┐
        │ Push to AWS S3│
        └──────┬────────┘
               ▼
        ┌──────────────┐
        │ FastAPI App  │
        └──────┬────────┘
               ▼
        ┌──────────────┐
        │  User Web UI │
        └──────────────┘


# ⚙️ Tech Stack

| Category | Tools                          |
| -------- | ------------------------------ |
| Language | Python 3.10                    |
| ML       | Scikit-learn, Imbalanced-learn |
| Backend  | FastAPI                        |
| Database | MongoDB Atlas                  |
| Cloud    | AWS S3, ECR, EC2               |
| DevOps   | Docker, GitHub Actions         |
| CI/CD    | Self-hosted Runner on EC2      |

---

# 📂 Project Structure

```
ML-Ops_proj1
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── entity/
│   │   ├── estimator.py
│   │   ├── s3_estimator.py
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── cloud_storage/
│   ├── configuration/
│   ├── utils/
│   ├── logger.py
│   └── exception.py
│
├── templates/
├── static/
├── notebook/
├── app.py
├── demo.py
├── Dockerfile
└── .github/workflows/aws.yaml
```

---

# 🔄 ML Pipeline Workflow

## 1️⃣ Data Ingestion

* Fetches data from **MongoDB Atlas**
* Splits into train/test
* Stores artifacts locally

## 2️⃣ Data Validation

* Validates schema from YAML
* Ensures feature consistency

## 3️⃣ Data Transformation

* Feature engineering
* Scaling (StandardScaler + MinMaxScaler)
* Handles class imbalance using **SMOTEENN**

## 4️⃣ Model Training

* Trains RandomForestClassifier
* Saves preprocessing + model together

## 5️⃣ Model Evaluation

* Compares with production model from S3
* Deploys only if performance improves

## 6️⃣ Model Pusher

* Pushes model to AWS S3 Model Registry

---

# ☁️ Cloud & MLOps Features

## Model Registry – AWS S3

* Versioned production models
* Automatic comparison with new models

🔄 CI/CD Pipeline

Every GitHub push automatically:

Developer Push
      │
      ▼
GitHub Actions Workflow
      │
      ▼
Build Docker Image
      │
      ▼
Push to AWS ECR
      │
      ▼
Deploy on EC2 Server
      │
      ▼
Live App Updated Automatically

# 🌐 Web Application

### FastAPI Features

* Train model from `/train`
* Predict from `/` UI form
* Loads model directly from S3

---

# 🐳 Run Locally

## 1️⃣ Create Environment

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

## 2️⃣ Setup Environment Variables

### MongoDB

```bash
export MONGODB_URL="your_mongodb_connection"
```

### AWS

```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
```

---

## 3️⃣ Run Training Pipeline

```bash
python demo.py
```

---

## 4️⃣ Run Web App

```bash
python app.py
```

Open:

```
http://localhost:5000
```

---

# 🐳 Docker Deployment

```bash
docker build -t vehicleproj .
docker run -p 5000:5000 vehicleproj
```

---

# 🚀 Deploy on AWS EC2

The project includes:

* Dockerfile
* GitHub Actions workflow
* Self-hosted runner setup

After deployment, access the app at:

```
http://<EC2-IP>:5080
```

---

# 🎯 Key MLOps Highlights

✔ End-to-End Automated ML Pipeline
✔ Model Versioning & Registry
✔ CI/CD Deployment
✔ Cloud Integration
✔ Production-ready API
✔ Real-time Predictions

---

## 👨‍💻 Author

**Adarsh Singh**

Crafted with ❤️
---

# ⭐ If you like this project

Give it a ⭐ on GitHub and share feedback!
