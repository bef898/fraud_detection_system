# fraud_detection_system
## A comprehensive fraud detection system for identifying and visualizing fraudulent transactions.
### Project Overview

This project builds an end-to-end fraud detection system with the following key features:

Data preparation and analysis for fraud insights.
Machine learning models for fraud prediction.
Explainability tools to make model predictions understandable.
API for serving real-time fraud predictions.
An interactive dashboard for monitoring fraud trends.
Table of Contents
Project Structure
Installation
Data Preparation and Model Training
Model Explainability
API and Deployment
Dashboard
Usage
Future Improvements


# Project Structure
fraud_detection_system/
├── data/                     # Folder containing fraud data CSV files
├── fraud_dashboard/          # Folder for Flask + Dash dashboard
│   ├── app.py                # Main Flask app
│   ├── dashboard.py          # Dash dashboard for visualization
│   └── templates/
│       └── index.html        # Main page template for Flask
├── model/                    # Folder to store trained models
│   └── model.pkl             # Serialized machine learning model
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation


# Installation
## Clone the Repository:

git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
### Set Up a Virtual Environment:
python -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
### Install Dependencies:
pip install -r requirements.txt
Prepare Data: Ensure you have the fraud_data.csv file in the data folder.

# Data Preparation and Model Training
### Task 1: Data Analysis and Preprocessing
Handle missing values, clean data, perform EDA, and engineer features for optimal performance.
Merge IP data to include geographic insights.
Normalize and encode data for machine learning compatibility.
### Task 2: Model Building and Training
Models Used: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP, CNN, RNN, and LSTM.
Experiment Tracking: MLflow is used to track experiments, log metrics, and manage model versions.
Results: Gradient Boosting and Random Forest showed the best performance, selected for deployment.

# Model Explainability
### Task 3: Model Explainability with SHAP and LIME
SHAP: Used for feature importance, force, and dependence plots to visualize global and local feature contributions.
LIME: Generates local explanations to make individual fraud predictions interpretable.
Run SHAP and LIME by following the instructions in the dashboard.py file to visualize feature importance and interpret predictions.

# API and Deployment
### Task 4: Flask API for Fraud Detection
# API Setup:

The Flask app (app.py) serves fraud predictions and summary data via endpoints.
Endpoints:
/api/summary: Provides summary statistics of fraud cases.
/api/fraud_over_time: Serves time-series data of fraud cases.
## Docker Deployment:

Build and run the Flask API within a Docker container:
docker build -t fraud-detection-api .
docker run -p 5000:5000 fraud-detection-api
Access the API at http://127.0.0.1:5000.

# Dashboard
## Task 5: Interactive Dashboard with Flask and Dash
## Dashboard Components:

Summary Boxes: Display total transactions, fraud cases, and fraud percentage.
Line Chart: Shows fraud trends over time.
Bar Chart: Compares fraud cases by device and browser type.
## Access the Dashboard:

The dashboard is integrated into Flask using Dash and accessible at http://127.0.0.1:5000.
Embedded within index.html, it provides real-time fraud monitoring.

## Usage
# Run the Flask API and Dashboard:

python fraud_dashboard/app.py
# Access the API and Dashboard:

API: Access endpoints such as http://127.0.0.1:5000/api/summary.
Dashboard: Access the dashboard at http://127.0.0.1:5000.
# Testing API Endpoints:

Test endpoints using Postman or cURL:

curl http://127.0.0.1:5000/api/summary
Docker Deployment:

To deploy in a Docker container, ensure Docker is installed and run the commands in the API and Deployment section.

### Future Improvements
Automated Model Retraining: Implement retraining to adapt to evolving fraud patterns.
Enhanced Scalability: Deploy on cloud infrastructure for larger-scale data.
Unsupervised Anomaly Detection: Add unsupervised models to detect novel fraud cases.
