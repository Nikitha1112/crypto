from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load tokenizers and model
key_tokenizer = load('key_tokenizer.joblib')
cipher_tokenizer = load('cipher_tokenizer.joblib')
model = load_model('final_model (2).h5')  # Replace with your actual model file path
label_encoder = load('label_encoder.joblib')  # Replace with your actual label encoder file path

# Define maxlen based on your training data (adjust if needed)
key_maxlen = 423  # Example value, adjust as per your trained data
cipher_maxlen = 512  # Example value, adjust as per your trained data

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload.html')
@app.route('/predict', methods=['POST'])

def predict():
    try:
        if request.is_json:
            data = request.get_json()
           
            key = data.get('key')
            cipher_text = data.get('cipher_text')
            key_length = data.get('key_length')
            cipher_length = data.get('cipher_length')

            # Ensure key_length and cipher_length are provided
            if key_length is None or cipher_length is None:
                return jsonify({"error": "Key length and cipher length must be provided"}), 400

            # Step 1: Tokenize the key and cipher text using the loaded tokenizers
            key_sequence = key_tokenizer.texts_to_sequences([key])
            key_padded = pad_sequences(key_sequence, maxlen=key_maxlen, padding='post')

            cipher_sequence = cipher_tokenizer.texts_to_sequences([cipher_text])
            cipher_padded = pad_sequences(cipher_sequence, maxlen=cipher_maxlen, padding='post')

            # Step 2: Make prediction using the model with 4 distinct inputs (key_padded, cipher_padded, key_length, cipher_length)
            prediction = model.predict([key_padded, cipher_padded, np.array([[key_length]]), np.array([[cipher_length]])])

            # Step 3: Decode the prediction (Top 3 algorithms)
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_algorithms = label_encoder.inverse_transform(top_3_indices)

            # Get the top predicted algorithm
            predicted_algorithm = label_encoder.inverse_transform([np.argmax(prediction, axis=1)])[0]

            # Return the prediction and top 3 algorithms
            return jsonify({'prediction': predicted_algorithm, 'top_3': top_3_algorithms.tolist()})

        else:
            return jsonify({"error": "Request must be in JSON format"}), 415

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

