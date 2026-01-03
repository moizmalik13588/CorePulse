from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
model = joblib.load("LR_heart_model.pkl")
columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Mapping all inputs from form
        raw_input = {
            'Age': int(data['age']),
            'RestingBP': int(data['rbp']),
            'Cholesterol': int(data['chol']),
            'FastingBS': int(data['fbs']),
            'MaxHR': int(data['mhr']),
            'Oldpeak': float(data['oldpeak']),
            f"Sex_{data['sex']}": 1,
            f"ChestPainType_{data['cp']}": 1,
            f"RestingECG_{data['ecg']}": 1,
            f"ExerciseAngina_{data['angina']}": 1,
            f"ST_Slope_{data['slope']}": 1
        }
        
        input_df = pd.DataFrame([raw_input])
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[columns]
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)