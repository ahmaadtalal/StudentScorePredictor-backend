from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model
with open("StudentScorePredictor_Model.pkl", "rb") as f:
    model = pickle.load(f)

# Default/mode values (same as your frontend defaults)
mode_values = {
    "Hours_Studied": 20,
    "Attendance": 67,
    "Parental_Involvement": 1,
    "Access_to_Resources": 1,
    "Extracurricular_Activities": 1,
    "Sleep_Hours": 7,
    "Previous_Scores": 66,
    "Motivation_Level": 1,
    "Internet_Access": 1,
    "Tutoring_Sessions": 1,
    "Family_Income": 0,
    "Teacher_Quality": 1,
    "School_Type": 0,
    "Peer_Influence": 2,
    "Physical_Activity": 3,
    "Learning_Disabilities": 0,
    "Parental_Education_Level": 0,
    "Distance_from_Home": 0,
    "Gender": 0
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Fill missing values with mode values
    input_features = []
    for key in mode_values.keys():
        value = data.get(key, mode_values[key])
        input_features.append(value)
    
    input_array = np.array([input_features])  # shape (1, 19)
    
    prediction = model.predict(input_array)[0]
    
    return jsonify({"predicted_score": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)
