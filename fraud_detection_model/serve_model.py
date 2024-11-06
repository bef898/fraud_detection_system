from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Fraud Detection Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    logging.info(f"Received data: {data}")

    # Convert data into numpy array for prediction
    input_data = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # Log the prediction result
    logging.info(f"Prediction: {prediction[0]}")
    
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
