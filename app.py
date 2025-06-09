from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('diabetes_model.pkl')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = [data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                  data['SkinThickness'], data['Insulin'], data['BMI'],
                  data['DiabetesPedigreeFunction'], data['Age']]
    prediction = model.predict([input_data])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
