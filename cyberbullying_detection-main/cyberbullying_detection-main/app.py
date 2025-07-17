from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import uuid
import speech_recognition as sr  # ⬅️ NEW IMPORT

app = Flask(__name__)
app.secret_key = 'your_secret_key'

basedir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(basedir, "stopwords.txt"), "r") as file:
    stopwords = file.read().splitlines()

vectorizer = pickle.load(open(os.path.join(basedir, "tfidfvectoizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(basedir, "LinearSVCTuned.pkl"), 'rb'))

users = {"admin": "password123"}  # Dummy login credentials
classification_history = []

# ✅ NEW FUNCTION: Voice recognition using microphone
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Speech recognition API unavailable."

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        classification_history.append(prediction)
        return redirect(url_for('stats', result=prediction))
    return render_template('index.html')

# ✅ NEW ROUTE: Handle voice input
@app.route('/voice-input', methods=['GET', 'POST'])
def voice_input():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        spoken_text = recognize_speech_from_mic()
        transformed_input = vectorizer.transform([spoken_text])
        prediction = model.predict(transformed_input)[0]
        classification_history.append(prediction)
        return render_template('result.html', text=spoken_text, result=prediction)
    return render_template('voice_input.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/stats')
def stats():
    bullying_count = classification_history.count(1)
    non_bullying_count = classification_history.count(0)

    labels = ['Non-Bullying', 'Bullying']
    values = [non_bullying_count, bullying_count]

    filename = f"static/chart_{uuid.uuid4().hex}.png"
    path = os.path.join(basedir, filename)

    plt.figure(figsize=(4, 4))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.savefig(path)
    plt.close()

    return render_template('stats.html', result=request.args.get('result'), chart=filename)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
