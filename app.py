import streamlit as st
import numpy as np
import librosa
import os
import sounddevice as sd
from scipy.io.wavfile import write

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# Title
# -------------------------------
st.title("🎤 Voice Emotion Recognition Dashboard")
st.write("Speak and detect your emotion using AI")

# -------------------------------
# Emotion Mapping (RAVDESS)
# -------------------------------
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_feature(file_name):
    audio, sample_rate = librosa.load(file_name, duration=3, offset=0.5)
    mfccs = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
        axis=0
    )
    return mfccs

# -------------------------------
# Load Dataset (DEBUG VERSION)
# -------------------------------
@st.cache_data
def load_data():
    X = []
    y = []

    # 🔥 CHANGE THIS PATH IF NEEDED
    dataset_path = r"C:\Users\sruth\Downloads\voiceemotionrecognition\dataset"

    st.write("📍 Looking dataset at:", dataset_path)

    if not os.path.exists(dataset_path):
        st.error(f"❌ Dataset folder NOT found!")
        return np.array(X), np.array(y)

    actors = os.listdir(dataset_path)
    st.write("📁 Actors found:", actors)

    for actor in actors:
        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        files = os.listdir(actor_path)
        st.write(f"📂 {actor} → {len(files)} files")

        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(actor_path, file)

                try:
                    feature = extract_feature(file_path)
                    X.append(feature)

                    parts = file.split("-")

                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        label = emotions.get(emotion_code, "neutral")
                    else:
                        label = "neutral"

                    y.append(label)

                except Exception as e:
                    st.warning(f"⚠️ Skipped: {file}")

    return np.array(X), np.array(y)

# -------------------------------
# Dataset Loading
# -------------------------------
st.subheader("📂 Dataset Loading")

if st.button("🔄 Reload Dataset"):
    st.cache_data.clear()

st.write("🚀 App started...")

X, y = load_data()

st.write("📊 Total Samples Loaded:", len(X))

# ❗ FIX: NO st.stop() → prevents blank page
if len(X) == 0:
    st.error("❌ Dataset not loading! Check path or files.")
else:
    st.success("✅ Dataset loaded successfully!")

    # -------------------------------
    # Train Model
    # -------------------------------
    st.subheader("🧠 Model Training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)

    st.success("✅ Model trained!")

    # -------------------------------
    # Evaluation
    # -------------------------------
    st.subheader("📊 Model Evaluation")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"🎯 Accuracy: {accuracy*100:.2f}%")

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    # -------------------------------
    # Record Voice
    # -------------------------------
    st.subheader("🎙️ Record Voice")

    duration = st.slider("Recording Duration (seconds)", 1, 5, 3)

    if st.button("Start Recording"):
        fs = 44100
        st.info("Recording... Speak now!")

        os.makedirs("recordings", exist_ok=True)

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        write("recordings/test.wav", fs, recording)
        st.success("✅ Recording completed!")

    # -------------------------------
    # Predict Emotion
    # -------------------------------
    st.subheader("🔍 Predict Emotion")

    if st.button("Analyze Emotion"):
        file_path = "recordings/test.wav"

        if not os.path.exists(file_path):
            st.error("⚠️ Please record audio first!")
        else:
            try:
                feature = extract_feature(file_path)

                prediction = model.predict([feature])[0]
                proba = model.predict_proba([feature])

                confidence = np.max(proba) * 100

                st.success(f"Emotion: {prediction}")
                st.info(f"Confidence: {confidence:.2f}%")

            except Exception as e:
                st.error(f"Error: {e}")