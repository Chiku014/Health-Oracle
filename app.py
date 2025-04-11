from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load models
MODEL_DIR = "models"

diabetes_model = pickle.load(open(os.path.join(MODEL_DIR, 'diabetes_prediction_model.pkl'), 'rb'))
cardiac_model = pickle.load(open(os.path.join(MODEL_DIR, 'heart_disease_model.pkl'), 'rb'))
obesity_model = pickle.load(open(os.path.join(MODEL_DIR, 'Obesity_Prediction_Model.pkl'), 'rb'))
obesity_encoder = pickle.load(open(os.path.join(MODEL_DIR, 'Obesity_Label_Encoder.pkl'), 'rb'))

# Route 1: Diabetes Prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json
    input_data = np.array([[data['age'], data['bmi'], data['glucose'], int(data['hypertension'])]])
    result = diabetes_model.predict(input_data)[0]
    return jsonify({'result': str(result)})

# Route 2: Cardiac Prediction
@app.route('/predict/cardiac', methods=['POST'])
def predict_cardiac():
    data = request.json
    features = np.array([[data['age'], data['cholesterol'], data['bp'], data['max_hr'], data['oldpeak']]])
    result = cardiac_model.predict(features)[0]
    return jsonify({'result': str(result)})

# Route 3: Obesity Prediction
@app.route('/predict/obesity', methods=['POST'])
def predict_obesity():
    data = request.json
    # Encode categorical inputs like Gender, Activity Level etc.
    gender = obesity_encoder['Gender'].transform([data['Gender']])[0]
    activity = obesity_encoder['CALC'].transform([data['CALC']])[0]

    input_data = np.array([[data['Age'], data['Height'], data['Weight'], gender, activity]])
    result = obesity_model.predict(input_data)[0]
    return jsonify({'result': str(result)})

@app.route('/')
def hello():
    return 'Health Oracle Flask API is live!'

if __name__ == '__main__':
    app.run(debug=True)
