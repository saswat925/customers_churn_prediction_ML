# Telecom Customer Churn Prediction

A production-style machine learning project for predicting telecom customer churn. This repository demonstrates an end-to-end ML workflow covering data cleaning, feature engineering, model benchmarking, serialization, and deployment through a FastAPI service.

## Overview

Customer churn is one of the most critical business metrics in telecom operations. This project builds a predictive model to estimate the probability that a customer will churn and classifies the risk level as Low, Medium, or High.

The solution includes:
- Data validation and anomaly handling
- Exploratory data analysis (EDA)
- Feature engineering and preprocessing
- Stratified train/test split
- Model comparison and selection
- Model export using joblib
- REST API deployment with FastAPI

## Project Structure

```text
customers_churn_prediction_ML/
├── customer_churn_predictions.ipynb
├── app.py
├── requirements.txt
├── README.md
├── telecom_churn.csv
├── telecom_customer_predictions.csv
├── models/
│   └── telecom_churn_model.joblib
└── .gitignore
```

## Dataset

The project uses a telecom churn dataset stored in:

- `telecom_churn.csv`

The dataset contains customer attributes such as:
- `customer_id`
- `telecom_partner`
- `gender`
- `age`
- `state`
- `city`
- `pincode`
- `date_of_registration`
- `num_dependents`
- `estimated_salary`
- `calls_made`
- `sms_sent`
- `data_used`
- `churn`

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- SciPy
- FastAPI
- Uvicorn
- Pydantic
- Joblib

## Machine Learning Workflow

The project follows a complete ML lifecycle:

1. Data ingestion and schema validation
2. Missing-value and anomaly detection
3. Data cleaning and standardization
4. Customer tenure feature generation
5. Categorical encoding
6. Train/test split with stratification
7. Model evaluation and benchmarking
8. Final model selection
9. Model serialization
10. Inference pipeline deployment via API

## Model

The final trained model is stored at:

- `models/telecom_churn_model.joblib`

The pipeline includes:
- numerical feature scaling
- one-hot encoded categorical variables
- Random Forest classifier

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/saswat925/customers_churn_prediction_ML.git
cd customers_churn_prediction_ML
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Open the notebook

```bash
jupyter notebook
```

Then open:

- `customer_churn_predictions.ipynb`

## FastAPI Deployment

Start the API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open the Swagger UI:

- http://localhost:8000/docs

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "Operational",
  "model_loaded": true
}
```

### Predict Churn Risk

```http
POST /predict
```

Request example:

```json
{
  "age": 35,
  "num_dependents": 2,
  "estimated_salary": 65000,
  "calls_made": 75,
  "sms_sent": 25,
  "data_used": 4000.0,
  "tenure_days": 520,
  "telecom_partner_BSNL": false,
  "telecom_partner_Reliance Jio": true,
  "telecom_partner_Vodafone": false,
  "gender_M": true
}
```

Example response:

```json
{
  "churn_probability": 0.34,
  "predicted_churn": 0,
  "risk_category": "Medium"
}
```

## Risk Classification

The model assigns a churn risk category based on probability:

- `Low`: probability < 0.3
- `Medium`: probability between 0.3 and 0.7
- `High`: probability >= 0.7

## Project Highlights

- End-to-end ML pipeline from raw data to deployment
- Production-ready model serving with FastAPI
- Portfolio-friendly project structure
- Real-world telecom churn use case
- Reproducible environment via `requirements.txt`

## License

This project is licensed under the MIT License unless otherwise stated.

## Author

Saswat Sahoo

## Contact

For collaboration or project inquiries, please visit:
- https://github.com/saswat925
