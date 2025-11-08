File Structure
Your project folder should look exactly like this:

/capstone-main
    |
    |--  app.py                   (The Python Flask backend server)
    |
    |-- ravdess_emotion_speaker_independent.keras  (The AI Model)
    |-- feature_scaler.pkl         (The AI Model's scaler)
    |-- label_encoder.pkl          (The AI Model's encoder)
    |
    |-- /templates
    |   |-- index.html             (The HTML frontend)
    |
    |-- /.venv                     (Your Python virtual environment folder)
    |
    |-- README.md                  (This file)
Setup and Run Instructions
Follow these steps exactly to run the project.

Step 1: Set Up the Python Environment
Open your project folder (capstone-main) in VS Code.

Open the VS Code Terminal ( Ctrl + ~ ).

Activate your virtual environment. This is the most important step.

Bash

.\.venv\Scripts\activate
Your terminal prompt should now start with (.venv).

Install the required libraries. With your virtual environment active, run this command:

Bash

pip install flask tensorflow scikit-learn librosa joblib numpy requests flask-cors soundfile
(This list includes soundfile to make sure librosa can read the .wav files correctly).

Step 3: Add Your API Key
Open the app.py file.

Find the line at the top (around line 22) that says:

Python

GEMINI_API_KEY = '' # <-- ⚠️ PASTE YOUR GEMINI API KEY HERE
Go to the Google AI Studio to get your free API key.

Paste your key between the quotes.

Step 4: Run the Application
Make sure your (.venv) is still active in the terminal.

Run the app:

Bash

python app.py
Your terminal will show:

* Running on http://127.0.0.1:5000
Open your web browser (like Chrome) and go to that address: http://127.0.0.1:5000