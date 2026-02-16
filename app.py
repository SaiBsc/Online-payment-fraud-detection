import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# 1. Initialize variables as None to prevent "not defined" errors
model = None
encoder = None

# 2. Use absolute paths to locate files in your project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

# 3. Enhanced Loading Logic
def load_assets():
    global model, encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            # joblib is more memory-efficient than pickle for large arrays
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(ENCODER_PATH)
            print("✅ Model and Encoder loaded successfully!")
        else:
            print(f"❌ Error: Files not found at {BASE_DIR}")
    except MemoryError:
        print("❌ MemoryError: The model file is too large for your RAM.")
    except Exception as e:
        print(f"❌ Error loading files: {e}")

# Load assets when the script starts
load_assets()

@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 4. Check if files were loaded before predicting
    if model is None or encoder is None:
        return "System Error: Model or Encoder is not loaded. Please check terminal logs."

    try:
        # Extract data from the form
        step = float(request.form.get("step", 0))
        tx_type = request.form.get("type")
        amount = float(request.form.get("amount", 0))
        oldbalanceOrg = float(request.form.get("oldbalanceOrg", 0))
        newbalanceOrig = float(request.form.get("newbalanceOrig", 0))
        oldbalanceDest = float(request.form.get("oldbalanceDest", 0))
        newbalanceDest = float(request.form.get("newbalanceDest", 0))

        # 5. Transform the transaction type using the loaded encoder
        tx_type_encoded = encoder.transform([tx_type])[0]

        # Arrange features in the exact order the model expects
        features = np.array([[
            step, tx_type_encoded, amount, 
            oldbalanceOrg, newbalanceOrig, 
            oldbalanceDest, newbalanceDest
        ]])

        # 6. Perform prediction
        prediction = model.predict(features)
        
        result = "Fraud Transaction ❌" if prediction[0] == 1 else "Safe Transaction ✅"

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == "__main__":
    # debug=True is helpful for development, but can double memory usage
    app.run(debug=True)