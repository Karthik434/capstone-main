import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests 
import traceback

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(app) 

# --- Global Variables & Model Loading ---
ML_MODEL_PATH = 'ravdess_emotion_speaker_independent.keras'
SCALER_PATH = 'feature_scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'
GEMINI_API_KEY = '' # <-- ⚠️ PASTE YOUR GEMINI API KEY HERE
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"

N_MELS = 128
MAX_PAD_LEN = 300
SR = 22050

try:
    print("Loading machine learning models...")
    ml_model = tf.keras.models.load_model(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model files: {e}")
    ml_model = None

# --- Audio Preprocessing Function (from Code 2) ---
def extract_logmel_for_prediction(file_path):
    """
    Processes a single audio file for prediction.
    Relies on librosa + soundfile to read the file.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SR) 
        audio = librosa.util.normalize(audio)
        
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512)
        logmel = librosa.power_to_db(mel, ref=np.max)
        
        if logmel.shape[1] > MAX_PAD_LEN:
            start = (logmel.shape[1] - MAX_PAD_LEN) // 2
            logmel = logmel[:, start:start + MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - logmel.shape[1]
            logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode='constant')
            
        return logmel.T.astype(np.float32)
    except Exception as e:
        print(f"--- ERROR in extract_logmel_for_prediction ---")
        print(f"File: {file_path}")
        print(f"Error: {e}")
        traceback.print_exc() 
        print("----------------------------------------------")
        return None

# --- Backend Routes ---

@app.route('/')
def index():
    if 'emotion' not in session:
        session['emotion'] = 'neutral'
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if ml_model is None:
        return jsonify({"error": "Emotion model is not loaded."}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = None
    try:
        # 1. Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 2. Process the file with librosa
        print(f"Processing {filepath}...")
        features = extract_logmel_for_prediction(filepath)
        
        # This will fail if the file is not a valid format (e.g., webm)
        # This app now only supports formats librosa+soundfile can read (like WAV)
        if features is None:
            return jsonify({"error": "Could not process audio. Please upload a valid WAV file."}), 500

        # 3. Scale and Predict
        features_2d = features.reshape(-1, features.shape[-1])
        features_scaled_2d = scaler.transform(features_2d)
        features_scaled = features_scaled_2d.reshape(features.shape)
        
        features_scaled = np.expand_dims(features_scaled, axis=0)

        prediction = ml_model.predict(features_scaled)
        pred_index = np.argmax(prediction, axis=1)[0]
        emotion = label_encoder.classes_[pred_index]

        # 4. Save emotion in session
        session['emotion'] = emotion
        
        # 5. Return success
        return jsonify({"success": True, "emotion": emotion})

    except Exception as e:
        print(f"Error during /upload_audio route: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 6. Clean up the uploaded file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


@app.route('/send_message', methods=['POST'])
def send_message():
    """Handles sending a chat message to the Gemini API"""
    data = request.json
    user_message = data.get('message', '')
    
    # Get the *CURRENT* emotion from the session
    emotion = session.get('emotion', 'neutral') # Default to 'neutral' if not set

    # --- This is the key part: The System Prompt ---
    # It dynamically uses the latest emotion detected
    system_prompt = f"""
    You are an AI assistant. You are talking to a user.
    The user's *current* emotional tone has been analyzed as: **{emotion}**.
    
    If the emotion is 'sad', 'angry', or 'fearful', be extra empathetic, supportive, and gentle.
    If the emotion is 'happy' or 'surprised', feel free to be more enthusiastic and match their energy.
    If the emotion is 'neutral' or 'calm', just be your normal helpful self.
    
    Respond to the user's message below.
    """
    
    # --- Call the Gemini API ---
    if not GEMINI_API_KEY:
        return jsonify({"response": "Error: Gemini API key is not set in the backend."})

    try:
        payload = {
            "contents": [{
                "parts": [{"text": user_message}]
            }],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=20)
        
        response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)
        
        result = response.json()
        
        if 'candidates' in result and result['candidates'][0]['content']['parts'][0]['text']:
            ai_response = result['candidates'][0]['content']['parts'][0]['text']
        else:
            ai_response = "Sorry, I couldn't get a response. (Empty candidate)"

    except requests.exceptions.RequestException as e:
        print(f"Gemini API Error: {e}")
        ai_response = f"Sorry, I encountered an error calling the AI. (Error: {e})"
    except Exception as e:
        print(f"Error processing Gemini response: {e}")
        ai_response = f"Sorry, I encountered an error. ({e})"

    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)