from flask import Flask, render_template, redirect, url_for, request
import subprocess
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import fft
from pydub import AudioSegment  # do konwersji webm → wav

app = Flask(__name__)
RECORD_DIR = "static/recordings"
RECORD_PATH = os.path.join(RECORD_DIR, "output.wav")
USER_UPLOAD_PATH = os.path.join(RECORD_DIR, "input_user.wav")

# ---------------------------- STRONY ----------------------------
@app.route("/")
def index():
    nagrany = os.path.exists(RECORD_PATH)
    status = request.args.get("status", "")
    return render_template("index.html", nagrany=nagrany, status=status)


@app.route("/nagraj", methods=["POST"])
def nagraj():
    os.makedirs(RECORD_DIR, exist_ok=True)
    subprocess.run([
        "arecord", "-D", "plughw:3,0", "-f", "cd", "-t", "wav", "-d", "5",
        "-r", "44100", RECORD_PATH
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


# ---------------------- STRONA: SKALA INTERAKTYWNA ----------------------
@app.route("/skala")
def skala():
    nuty = [
        ("C3", 130.81), ("C#3", 138.59), ("D3", 146.83), ("D#3", 155.56), ("E3", 164.81),
        ("F3", 174.61), ("F#3", 185.00), ("G3", 196.00), ("G#3", 207.65), ("A3", 220.00),
        ("A#3", 233.08), ("H3", 246.94), ("C4", 261.63), ("C#4", 277.18), ("D4", 293.66),
        ("D#4", 311.13), ("E4", 329.63), ("F4", 349.23), ("F#4", 369.99), ("G4", 392.00),
        ("G#4", 415.30), ("A4", 440.00), ("A#4", 466.16), ("H4", 493.88), ("C5", 523.25),
        ("C#5", 554.37), ("D5", 587.33), ("D#5", 622.25), ("E5", 659.25), ("F5", 698.46),
        ("F#5", 739.99), ("G5", 783.99), ("G#5", 830.61), ("A5", 880.00), ("A#5", 932.33),
        ("H5", 987.77), ("C6", 1046.50)
    ]
    return render_template("skala.html", nuty=nuty)


@app.route("/analizuj_skale_manualnie", methods=["POST"])
def analizuj_skale_manualnie():
    plec = request.form.get("plec")
    niska_hz = float(request.form.get("nuta_niska"))
    wysoka_hz = float(request.form.get("nuta_wysoka"))

    # Znajdź odpowiadające nazwy nut
    nuty = [
        ("C3", 130.81), ("C#3", 138.59), ("D3", 146.83), ("D#3", 155.56), ("E3", 164.81),
        ("F3", 174.61), ("F#3", 185.00), ("G3", 196.00), ("G#3", 207.65), ("A3", 220.00),
        ("A#3", 233.08), ("H3", 246.94), ("C4", 261.63), ("C#4", 277.18), ("D4", 293.66),
        ("D#4", 311.13), ("E4", 329.63), ("F4", 349.23), ("F#4", 369.99), ("G4", 392.00),
        ("G#4", 415.30), ("A4", 440.00), ("A#4", 466.16), ("H4", 493.88), ("C5", 523.25),
        ("C#5", 554.37), ("D5", 587.33), ("D#5", 622.25), ("E5", 659.25), ("F5", 698.46),
        ("F#5", 739.99), ("G5", 783.99), ("G#5", 830.61), ("A5", 880.00), ("A#5", 932.33),
        ("H5", 987.77), ("C6", 1046.50)
    ]
    name_low = next((n[0] for n in nuty if abs(n[1] - niska_hz) < 1), "?")
    name_high = next((n[0] for n in nuty if abs(n[1] - wysoka_hz) < 1), "?")

    zakres = round(wysoka_hz - niska_hz, 2)

    # Określenie typu głosu
    typ_glosu = okresl_typ_glosu(plec, niska_hz, wysoka_hz)

    wynik = {
        "niska_hz": niska_hz,
        "wysoka_hz": wysoka_hz,
        "nuta_niska": name_low,
        "nuta_wysoka": name_high,
        "zakres": zakres,
        "typ_glosu": typ_glosu
    }

    return render_template("skala.html", wynik=wynik, nuty=nuty)


def okresl_typ_glosu(plec, low, high):
    if plec == "kobieta":
        if low >= 220 and high >= 880:
            return "Sopran"
        elif low >= 196 and high >= 784:
            return "Mezzosopran"
        elif low >= 164 and high >= 698:
            return "Alt"
        else:
            return "Nieokreślony (niski zakres dla kobiety)"
    else:  # mężczyzna
        if low >= 82 and high >= 330:
            return "Tenor"
        elif low >= 98 and high >= 392:
            return "Baryton"
        elif low >= 65 and high >= 260:
            return "Bas"
        else:
            return "Nieokreślony (niski zakres dla mężczyzny)"


# ---------------------- ANALIZA: PARAMETRY ----------------------
@app.route("/analizuj_parametry", methods=["POST"])
def analizuj_parametry():
    if "audio" not in request.files:
        return "Brak pliku audio", 400

    audio_file = request.files["audio"]
    os.makedirs(RECORD_DIR, exist_ok=True)
    user_path = os.path.join(RECORD_DIR, "user_param.wav")
    audio_file.save(user_path)

    ref_path = os.path.join("static", "reference", "ref.wav.wav")

    try:
        wynik = analiza_parametrow(ref_path, user_path)
        return render_template("parametry.html", analiza=wynik)
    except Exception as e:
        return f"Błąd analizy: {str(e)}", 500


def analiza_parametrow(ref_path, user_path):
    """
    Analiza śpiewu – metoda ATSIP (pełna wersja, 7 parametrów)
    """

    import numpy as np
    import librosa

    # --- Wczytanie audio ---
    y_ref, sr_ref = librosa.load(ref_path, sr=None)
    y_usr, sr_usr = librosa.load(user_path, sr=None)

    # --- INTONATION ---
    try:
        pitches_ref, mags_ref = librosa.piptrack(y=y_ref, sr=sr_ref)
        pitches_usr, mags_usr = librosa.piptrack(y=y_usr, sr=sr_usr)
        pitch_ref_vals = pitches_ref[mags_ref > np.median(mags_ref)]
        pitch_usr_vals = pitches_usr[mags_usr > np.median(mags_usr)]
        pitch_ref = float(np.median(pitch_ref_vals)) if len(pitch_ref_vals) > 0 else 0.0
        pitch_usr = float(np.median(pitch_usr_vals)) if len(pitch_usr_vals) > 0 else 0.0
    except Exception:
        pitch_ref = pitch_usr = 0.0
    Intonation = max(0, 1 - abs(pitch_ref - pitch_usr) / 100)

    # --- RHYTHM ---
    try:
        tempo_ref, _ = librosa.beat.beat_track(y=y_ref, sr=sr_ref)
        tempo_usr, _ = librosa.beat.beat_track(y=y_usr, sr=sr_usr)
        tempo_ref, tempo_usr = float(tempo_ref), float(tempo_usr)
    except Exception:
        tempo_ref = tempo_usr = 0.0
    Rhythm = max(0, 1 - abs(tempo_ref - tempo_usr) / 60)

    # --- VIBRATO ---
    def vibrato_strength(y, sr):
        try:
            pitches, mags = librosa.piptrack(y=y, sr=sr)
            mean_pitch = np.mean(pitches, axis=0)
            mean_pitch = mean_pitch[mean_pitch > 0]
            if len(mean_pitch) < 5:
                return 0.0
            return float(np.std(mean_pitch) / 30)
        except Exception:
            return 0.0
    vib_ref = vibrato_strength(y_ref, sr_ref)
    vib_usr = vibrato_strength(y_usr, sr_usr)
    Vibrato = max(0, 1 - abs(vib_ref - vib_usr))

    # --- VOLUME ---
    try:
        rms_ref = float(np.mean(librosa.feature.rms(y=y_ref)))
        rms_usr = float(np.mean(librosa.feature.rms(y=y_usr)))
    except Exception:
        rms_ref = rms_usr = 0.0
    Volume = max(0, 1 - abs(rms_ref - rms_usr) / 0.1)

    # --- VOICE QUALITY ---
    try:
        S_ref = np.abs(librosa.stft(y_ref))
        S_usr = np.abs(librosa.stft(y_usr))
        harm_ref, per_ref = librosa.decompose.hpss(S_ref)
        harm_usr, per_usr = librosa.decompose.hpss(S_usr)
        vq_ref = float(np.sum(harm_ref) / (np.sum(harm_ref) + np.sum(per_ref) + 1e-10))
        vq_usr = float(np.sum(harm_usr) / (np.sum(harm_usr) + np.sum(per_usr) + 1e-10))
    except Exception:
        vq_ref = vq_usr = 0.0
    VoiceQuality = max(0, 1 - abs(vq_ref - vq_usr))

    # --- PRONUNCIATION ---
    try:
        mfcc_ref = np.mean(librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13), axis=1)
        mfcc_usr = np.mean(librosa.feature.mfcc(y=y_usr, sr=sr_usr, n_mfcc=13), axis=1)
        diff = float(np.mean(np.abs(mfcc_ref - mfcc_usr)))
    except Exception:
        diff = 100.0
    Pronunciation = max(0, 1 - diff / 100)

    # --- PITCH DYNAMIC RANGE ---
    def pitch_range(y, sr):
        try:
            pitches, mags = librosa.piptrack(y=y, sr=sr)
            vals = pitches[mags > np.median(mags)]
            if len(vals) == 0:
                return 0.0
            return float(np.max(vals) - np.min(vals))
        except Exception:
            return 0.0
    pdr_ref = pitch_range(y_ref, sr_ref)
    pdr_usr = pitch_range(y_usr, sr_usr)
    PitchDynamicRange = max(0, 1 - abs(pdr_ref - pdr_usr) / 300)

    # --- OCENA OGÓLNA ---
    Overall = np.mean([
        Intonation, Rhythm, Vibrato,
        Volume, VoiceQuality, Pronunciation,
        PitchDynamicRange
    ]) * 100.0

    # --- WYNIKI ---
    wynik = {
        "🎵 Intonation": f"{Intonation * 100:.2f} %",
        "⏱️ Rhythm": f"{Rhythm * 100:.2f} %",
        "🌊 Vibrato": f"{Vibrato * 100:.2f} %",
        "🔊 Volume": f"{Volume * 100:.2f} %",
        "🧬 Voice Quality": f"{VoiceQuality * 100:.2f} %",
        "🗣️ Pronunciation": f"{Pronunciation * 100:.2f} %",
        "🔁 Pitch Dynamic Range": f"{PitchDynamicRange * 100:.2f} %",
        "⭐ Overall": f"{Overall:.2f} / 100"
    }

    return wynik


# ---------------------- ANALIZA: PARAMETRY 2 ----------------------
@app.route("/analizuj_parametry2", methods=["POST"])
def analizuj_parametry2():
    if "audio" not in request.files:
        return "Brak pliku audio", 400

    audio_file = request.files["audio"]
    target_freq = float(request.form.get("target_freq", 0.0))

    os.makedirs(RECORD_DIR, exist_ok=True)
    webm_path = os.path.join(RECORD_DIR, "parametry2_input.webm")
    wav_path = os.path.join(RECORD_DIR, "parametry2_input.wav")
    audio_file.save(webm_path)

    try:
        AudioSegment.from_file(webm_path, format="webm").export(wav_path, format="wav")
        analiza, feedback, wykres_path = analiza_czystosci(wav_path, target_freq)
        return render_template("parametry2.html", analiza=analiza, feedback=feedback, wykres_path=wykres_path)
    except Exception as e:
        return f"Błąd analizy: {str(e)}", 500


# ---------------------- ANALIZA: CZYSTOŚĆ ----------------------
def analiza_czystosci(path, target_freq):
    data, sample_rate = sf.read(path)
    if data.ndim == 2:
        data = data[:, 0]

    window_size = 8192
    window = data[:window_size] * np.hanning(window_size)

    spectrum = np.abs(fft(window))[:window_size // 2]
    freqs = np.fft.fftfreq(window_size, 1 / sample_rate)[:window_size // 2]

    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    diff = peak_freq - target_freq
    czystosc = max(0, 100 - (abs(diff) / target_freq) * 100)

    if abs(diff) < 5:
        feedback = "🌟 Idealnie czysto!"
    elif diff > 0:
        feedback = f"🔺 Za wysoko o {diff:.2f} Hz"
    else:
        feedback = f"🔻 Za nisko o {abs(diff):.2f} Hz"

    plt.figure(figsize=(12, 5))
    plt.plot(freqs, spectrum, color='blue', label='Zaśpiewany ton (FFT)')
    plt.axvline(x=peak_freq, color='black', linestyle='--', label=f'Peak: {peak_freq:.2f} Hz')
    plt.axvline(x=target_freq, color='red', linestyle='--', label=f'Wzorzec: {target_freq:.2f} Hz')
    plt.xlim(0, 1000)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title("Analiza czystości tonu (FFT)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    wykres_file = "czystosc_wykres.png"
    wykres_path = os.path.join("static", wykres_file)
    plt.savefig(wykres_path)
    plt.close()

    analiza = {
        "Dominująca częstotliwość": f"{peak_freq:.2f} Hz",
        "Wzorzec": f"{target_freq:.2f} Hz",
        "Różnica": f"{diff:+.2f} Hz",
        "Czystość tonu": f"{czystosc:.2f} %"
    }

    return analiza, feedback, wykres_file


# ---------------------- ANALIZA: LIBROSA ----------------------
def analiza_dzwieku(path, target_freq, ref_duration=1.5, ref_rms=0.05):
    y, sr = librosa.load(path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch = np.median(pitch_values) if len(pitch_values) > 0 else 0.0
    pitch_error = abs(pitch - target_freq)
    pitch_accuracy = max(0, 1 - pitch_error / 50)

    pitch_stability = max(0, 1 - np.std(pitch_values) / 30) if len(pitch_values) > 1 else 0.0
    rms = np.mean(librosa.feature.rms(y=y))
    loudness = min(1.0, rms / ref_rms)
    duration = librosa.get_duration(y=y, sr=sr)
    duration_score = min(1.0, duration / ref_duration)

    score = round((0.4 * pitch_accuracy + 0.2 * pitch_stability +
                   0.2 * loudness + 0.2 * duration_score) * 100, 2)

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


# ---------------------------- START ----------------------------
if __name__ == "__main__":
    os.makedirs(RECORD_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
