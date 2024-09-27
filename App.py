from flask import Flask, request, render_template, redirect, url_for, session, flash
import os
import numpy as np
import sqlite3
import librosa
from pydub import AudioSegment
from scipy.spatial.distance import euclidean, cosine
import pickle
import pyaudio
import wave

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DATABASE = 'app_database.db'
DEFAULT_BALANCE = 1000.00  # Default balance for new accounts

def create_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        dob TEXT NOT NULL,
        account_number TEXT UNIQUE NOT NULL,
        mfcc_features BLOB NOT NULL,
        balance REAL NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_path):
    """Convert non-WAV files to WAV format using pydub."""
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format='wav')
    return wav_path

def extract_mfcc(audio_path, n_mfcc=13):
    try:
        if not audio_path.lower().endswith('.wav'):
            audio_path = convert_to_wav(audio_path)
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

def verify_voice(features_1, features_2, method='euclidean', threshold=10):
    if method == 'euclidean':
        distance = euclidean(features_1, features_2)
        return "Voice Verified" if distance < threshold else "Voice Not Verified"
    elif method == 'cosine':
        similarity = 1 - cosine(features_1, features_2)
        return "Voice Verified" if similarity > 1 - threshold else "Voice Not Verified"
    else:
        raise ValueError("Unsupported method. Use 'euclidean' or 'cosine'.")

@app.route('/')
def index():
    return '''
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            color: #333;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 600px;
            margin: 20px auto;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        h1 {
            color: #004d99;
            margin-bottom: 20px;
        }
        a {
            display: block;
            margin: 10px;
            padding: 10px;
            background-color: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        a:hover {
            background-color: #0056b3;
        }
    </style>
    </head>
    <body>
    <div class="container">
        <h1>Welcome to DomBank</h1>
        <a href="/signup">Sign Up</a>
        <a href="/login">Login</a>
    </div>
    </body>
    </html>
    '''

@app.route('/signup')
def signup_page():
    return '''
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #004d99;
            text-align: center;
            margin-bottom: 20px;
        }
        label {
            font-size: 16px;
            margin-bottom: 8px;
            color: #0056b3;
        }
        input[type="text"], input[type="email"], input[type="date"], button {
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: calc(100% - 22px);
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #0056b3;
        }
        #status, #result {
            margin-top: 20px;
            text-align: center;
            font-size: 18px;
        }
        a {
            display: block;
            margin-top: 20px;
            text-align: center;
            color: #007bff;
            text-decoration: none;
            font-size: 16px;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    <script>
        let mediaRecorder;
        let audioChunks = [];

        async function startRecording() {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('voice_sample', audioBlob, 'voice_sample.wav');
                formData.append('full_name', document.getElementById('full_name').value);
                formData.append('email', document.getElementById('email').value);
                formData.append('dob', document.getElementById('dob').value);
                formData.append('account_number', document.getElementById('account_number').value);

                fetch('/signup', {
                    method: 'POST',
                    body: formData
                }).then(response => response.text())
                  .then(text => document.getElementById('result').innerText = text);
            };

            mediaRecorder.start();
            document.getElementById('status').innerText = "Recording...";
        }

        function stopRecording() {
            mediaRecorder.stop();
            document.getElementById('status').innerText = "Stopped Recording.";
        }
    </script>
    </head>
    <body>
    <div class="container">
        <h1>Sign Up</h1>
        <form id="signupForm" method="post" enctype="multipart/form-data">
            <label for="full_name">Full Name:</label>
            <input type="text" id="full_name" name="full_name" required>
            
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required>
            
            <label for="dob">Date of Birth:</label>
            <input type="date" id="dob" name="dob" required>
            
            <label for="account_number">Bank Account Number:</label>
            <input type="text" id="account_number" name="account_number" required>
            
            <button type="button" onclick="startRecording()">Start Recording</button>
            <button type="button" onclick="stopRecording()">Stop Recording</button>
            
            <p id="status">Status: Idle</p>
            <p id="result"></p>
        </form>
        <a href="/">Back to Home</a>
    </div>
    </body>
    </html>
    '''

@app.route('/login')
def login_page():
    return '''
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #004d99;
            text-align: center;
            margin-bottom: 20px;
        }
        label {
            font-size: 16px;
            margin-bottom: 8px;
            color: #0056b3;
        }
        input[type="email"], button {
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: calc(100% - 22px);
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #0056b3;
        }
        #status, #result {
            margin-top: 20px;
            text-align: center;
            font-size: 18px;
        }
        a {
            display: block;
            margin-top: 20px;
            text-align: center;
            color: #007bff;
            text-decoration: none;
            font-size: 16px;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    <script>
        let mediaRecorder;
        let audioChunks = [];

        async function startRecording() {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('voice_sample', audioBlob, 'voice_sample.wav');
                formData.append('email', document.getElementById('email').value);

                fetch('/login', {
                    method: 'POST',
                    body: formData
                }).then(response => response.text())
                  .then(text => document.getElementById('result').innerText = text);
            };

            mediaRecorder.start();
            document.getElementById('status').innerText = "Recording...";
        }

        function stopRecording() {
            mediaRecorder.stop();
            document.getElementById('status').innerText = "Stopped Recording.";
        }
    </script>
    </head>
    <body>
    <div class="container">
        <h1>Login</h1>
        <form id="loginForm" method="post" enctype="multipart/form-data">
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required>
            
            <button type="button" onclick="startRecording()">Start Recording</button>
            <button type="button" onclick="stopRecording()">Stop Recording</button>
            
            <p id="status">Status: Idle</p>
            <p id="result"></p>
        </form>
        <a href="/">Back to Home</a>
    </div>
    </body>
    </html>
    '''

@app.route('/signup', methods=['POST'])
def signup():
    full_name = request.form['full_name']
    email = request.form['email']
    dob = request.form['dob']
    account_number = request.form['account_number']

    if 'voice_sample' in request.files:
        voice_sample = request.files['voice_sample']
        if voice_sample and allowed_file(voice_sample.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'signup_' + email + '.wav')
            voice_sample.save(filename)
            features = extract_mfcc(filename)
            if features is None:
                return 'Error processing voice sample', 400
            mfcc_features = pickle.dumps(features)
        else:
            return 'Invalid file format', 400
    else:
        return 'No voice sample provided', 400

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT INTO customers (full_name, email, dob, account_number, mfcc_features, balance)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (full_name, email, dob, account_number, mfcc_features, DEFAULT_BALANCE))
        conn.commit()
        conn.close()
        return 'Sign up successful! You can now <a href="/login">login</a>.'
    except sqlite3.IntegrityError:
        conn.close()
        return 'Email or account number already exists.', 400

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']

    if 'voice_sample' in request.files:
        voice_sample = request.files['voice_sample']
        if voice_sample and allowed_file(voice_sample.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'login_' + email + '.wav')
            voice_sample.save(filename)
            features = extract_mfcc(filename)
            if features is None:
                return 'Error processing voice sample', 400

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT mfcc_features FROM customers WHERE email = ?', (email,))
            row = cursor.fetchone()
            conn.close()

            if row:
                stored_features = pickle.loads(row[0])
                result = verify_voice(features, stored_features)
                return result
            else:
                return 'User not found', 404
        else:
            return 'Invalid file format', 400
    else:
        return 'No voice sample provided', 400

if __name__ == '__main__':
    create_database()
    app.run(debug=True)

