from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pickle
import os

from flask import send_from_directory

app = Flask(__name__)

# Load the pre-trained model
model_path = "outcome_prediction_model1.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
# Load Model 2
with open("high_risk_model.pkl", "rb") as f2:
    high_risk_model = pickle.load(f2)


wards = [
    "Banjara Hills", "Begumpet", "Charminar", "Gachibowli", "Jubilee Hills",
    "Kukatpally", "LB Nagar", "Malakpet", "Mehdipatnam", "Musheerabad",
    "Quthbullapur", "Secunderabad", "Serilingampally", "Uppal"
]

symptoms_list = [
    "Fever", "joint pain", "muscle pain", "headache", "retro-orbital pain",
    "chills", "bleeding", "fatigue", "rash", "nausea"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-outcome')
def predict_outcome():
    return render_template('model1_prediction.html', wards=wards, symptoms=symptoms_list)

@app.route('/predict-high-risk')
def predict_high_risk():
    return render_template('model2_prediction.html', wards=wards, symptoms=symptoms_list)

@app.route('/predict-ward-demand')
def predict_ward_demand():
    return render_template('model3_ward_demand.html', wards=wards)

@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    data = request.json

    try:
        print("Received data:", data)

        age = data.get('age')
        gender = data.get('gender')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        rainfall = data.get('rainfall')
        platelet_count = data.get('platelet_count')
        month = data.get('month')
        year = data.get('year')
        age_group = data.get('age_group')
        ward = data.get('ward')
        symptoms = data.get('symptoms', [])

        gender_encoded = 0 if gender == 'Male' else 1
        age_group_map = {'Child': 0, 'Young Adult': 1, 'Adult': 2, 'Senior': 3}
        age_group_encoded = age_group_map.get(age_group, 0)
        year_map = {2022: 0, 2023: 1, 2024: 2, 2025: 3}
        year_encoded = year_map.get(year, 0)
        ward_encoded = [1 if w == ward else 0 for w in wards]
        symptoms_encoded = [1 if symptom in symptoms else 0 for symptom in symptoms_list]

        feature_vector = [
            age,
            gender_encoded,
            temperature,
            humidity,
            rainfall,
            platelet_count,
            month,
            year_encoded,
            age_group_encoded
        ] + ward_encoded + symptoms_encoded

        print("Feature vector length:", len(feature_vector))
        print("Feature vector:", feature_vector)

        features_np = np.array(feature_vector).reshape(1, -1)
        prediction = model.predict(features_np)[0]

        label_map = {
            0: "Critical",
            1: "Dead",
            2: "Hospitalized",
            3: "Recovered"
        }
        outcome_label = label_map.get(int(prediction), "Unknown")

        return jsonify({'outcome': outcome_label})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 400
    
@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    data = request.json

    try:
        age = data.get('age')
        gender = data.get('gender')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        rainfall = data.get('rainfall')
        platelet_count = data.get('platelet_count')
        month = data.get('month')
        year = data.get('year')
        age_group = data.get('age_group')
        ward = data.get('ward')
        symptoms = data.get('symptoms', [])

        # Same encodings as model 1
        gender_encoded = 0 if gender == 'Male' else 1
        age_group_map = {'Child': 0, 'Young Adult': 1, 'Adult': 2, 'Senior': 3}
        age_group_encoded = age_group_map.get(age_group, 0)
        year_map = {2022: 0, 2023: 1, 2024: 2, 2025: 3}
        year_encoded = year_map.get(year, 0)
        ward_encoded = [1 if w == ward else 0 for w in wards]
        symptoms_encoded = [1 if s in symptoms else 0 for s in symptoms_list]

        feature_vector = [
            age,
            gender_encoded,
            temperature,
            humidity,
            rainfall,
            platelet_count,
            month,
            year_encoded,
            age_group_encoded
        ] + ward_encoded + symptoms_encoded

        features_np = np.array(feature_vector).reshape(1, -1)
        prediction = high_risk_model.predict(features_np)[0]

        return jsonify({'high_risk': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_model3', methods=['POST'])
def predict_model3():
    try:
        data = request.json
        # TODO: Load model, validate/process input, predict demand level
        # For now, return a dummy response
        return jsonify({'demand_level': 'High'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/insights')
def insights():
    # Serve the insights page embedding the PDF
    return render_template('insights.html')

@app.route('/competitive-research')
def competitive_research():
    # Serve the competitive research PDF page
    return render_template('competitive_research.html')

@app.route('/download-pbix')
def download_pbix():
    return send_file('DengueDataReport.pbix', as_attachment=True)

# Ensure the PDF is accessible via static route
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
