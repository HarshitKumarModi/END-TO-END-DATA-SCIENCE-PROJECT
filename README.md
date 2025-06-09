# END-TO-END-DATA-SCIENCE-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: HARSHIT KUMAR MODI

INTERN ID: CT04DN455

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION OF THE PROJECT:

END-TO-END DATA SCIENCE PROECT

📌 Overview
This project is a complete end-to-end data science pipeline, from data preprocessing and model training to API deployment using Flask. It focuses on predicting whether a patient is likely to have diabetes based on medical diagnostic features. The solution includes:

Data Cleaning and Preprocessing

Model Training using Random Forest Classifier

Model Serialization (.pkl file)

API Development with Flask

Optional Web Deployment using Render

This project showcases the practical workflow of a real-world machine learning application—from raw data to a production-ready prediction API.

📈 Problem Statement
Build a machine learning model to predict the onset of diabetes based on diagnostic features (e.g., glucose level, blood pressure, BMI, etc.), and serve this model through a REST API that can be consumed by frontend applications, mobile apps, or other systems.

⚙️ Pipeline Workflow
1. 📊 Data Preprocessing
Handle missing values

Standardize numerical features

Apply one-hot encoding to categorical variables (if any)

Split data into training and testing sets

2. 🤖 Model Building
Use RandomForestClassifier from scikit-learn

Train the model on the cleaned dataset

Evaluate model accuracy

Save the trained model using joblib

3. 🚀 API Development
Build a Flask API with a /predict endpoint

Accept JSON input with 8 patient features:

Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Return a prediction (0: No diabetes, 1: Diabetes)

4. ☁️ Optional Deployment
Host the API on Render.com

Publicly accessible via a POST request

✅ Features
End-to-end ML pipeline: preprocessing → modeling → API

Modular and easy-to-read Python code

Model ready for production use

Clean and documented API for prediction

Deployable on cloud platforms like Render or Heroku

🧑‍💻 Technologies Used
Python 3.x

pandas – Data manipulation

scikit-learn – Machine learning & preprocessing

joblib – Model serialization

Flask – Web API framework

Render – (Optional) Free cloud deployment

OUTPUT:

