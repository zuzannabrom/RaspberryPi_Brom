from flask import Flask, render_template, redirect, url_for, request
import subprocess
import os
import numpy as np
import librosa

app = Flask(__name__)
RECORD_DIR = "static/recordings"
RECORD_PATH = os.path.join(RECORD_DIR, "output.wav")
USER_UPLOAD_PATH = os.path.join(RECORD_DIR, "input_user.wav")


@app.route("/")
def index():
    nagrany = os.path.exists(RECORD_PATH)
    status = request.args.get("status", "")
    return render_template("index.html", nagrany=nagrany, status=status)


@app.route("/nagraj", methods=["POST"])
def nagraj():
    os.makedirs(RECORD_DIR, exist_ok=True)
    subprocess.run([
        "arecord",
        "-D", "plughw:3,0",
        "-f", "cd",
        "-t", "wav",
        "-d", "5",
        "-r", "44100",
        RECORD_PATH
    ])
    return redirect(url_for("index", status="done"))


@app.route("/usun", methods=["POST"])
def usun():
    if os.path.exists(RECORD_PATH):
        os.remove(RECORD_PATH)
    return redirect(url_for("index"))


@app.route("/parametry")
def parametry():
    return render_template("parametry.html")


@app.route("/parametry2")
def parametry2():
    return render_template("parametry2.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" in request.files:
        audio_file = request.files["audio"]
        target_freq = float(request.form.get("target_freq", 0.0))
        # Zapisz nagranie użytkownika
        os.makedirs(RECORD_DIR, exist_ok=True)
        audio_file.save(USER_UPLOAD_PATH)
        # Analiza dźwięku
        wynik = analiza_dzwieku(USER_UPLOAD_PATH, target_freq)
        return render_template("parametry.html", analiza=wynik)
    return "Nie przesłano pliku", 400


def analiza_dzwieku(path, target_freq, ref_duration=1.5, ref_rms=0.05):
    try:
        y, sr = librosa.load(path)

        # 1. Dokładność intonacji (pitch accuracy)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch = np.median(pitch_values) if len(pitch_values) > 0 else 0.0
        pitch_error = abs(pitch - target_freq)
        pitch_accuracy = max(0, 1 - pitch_error / 50)  # zakładamy max błąd = 50 Hz

        # 2. Stabilność wysokości (pitch stability)
        if len(pitch_values) > 1:
            pitch_stability = max(0, 1 - np.std(pitch_values) / 30)  # zakładamy max odchylenie 30 Hz
        else:
            pitch_stability = 0.0
        # 3. Głośność (RMS loudness)
        rms = np.mean(librosa.feature.rms(y=y))
        loudness = min(1.0, rms / ref_rms)
        # 4. Czas trwania
        duration = librosa.get_duration(y=y, sr=sr)
        duration_score = min(1.0, duration / ref_duration)
        # Wagi
        w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
        # Ostateczna ocena
        score = round((w1 * pitch_accuracy + w2 * pitch_stability + w3 * loudness + w4 * duration_score) * 100, 2)
        return {
            "Dominująca częstotliwość (pitch)": f"{pitch:.2f} Hz",
            "Oczekiwana częstotliwość": f"{target_freq:.2f} Hz",
            "Błąd częstotliwości": f"{pitch_error:.2f} Hz",
            "Czystość intonacji": f"{pitch_accuracy:.2f}",
            "Stabilność wysokości": f"{pitch_stability:.2f}",
            "Głośność (RMS)": f"{rms:.4f}",
            "Znormalizowana głośność": f"{loudness:.2f}",
            "Czas trwania": f"{duration:.2f} s",
            "Znormalizowany czas trwania": f"{duration_score:.2f}",
            "Ocena końcowa": f"{score}/100"
        }

    except Exception as e:
        return {"Błąd": str(e)}
