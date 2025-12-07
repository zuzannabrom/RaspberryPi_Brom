from flask import Flask, render_template, redirect, url_for, request
import subprocess
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import fft
from pydub import AudioSegment  # do konwersji webm → wav
from flask import jsonify # do sprawdzania skali - nuty
import matplotlib.pyplot as plt # to do rysowania plotów w parametry - jeszcze nie ma dodanej tej opcji

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
        ("E2", 82.41),
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
        ("E2", 82.41),
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


@app.route("/sprawdz_nute", methods=["POST"])
def sprawdz_nute():
    if "audio" not in request.files:
        return jsonify({"feedback": "Brak pliku audio!"}), 400

    audio_file = request.files["audio"]
    target_freq = float(request.form.get("target_freq", 0.0))

    os.makedirs(RECORD_DIR, exist_ok=True)
    webm_path = os.path.join(RECORD_DIR, "check_note_input.webm")
    wav_path = os.path.join(RECORD_DIR, "check_note_input.wav")
    audio_file.save(webm_path)

    try:
        AudioSegment.from_file(webm_path, format="webm").export(wav_path, format="wav")
        _, feedback, _ = analiza_czystosci(wav_path, target_freq)
        return jsonify({"feedback": feedback})
    except Exception as e:
        return jsonify({"feedback": f"Błąd: {str(e)}"}), 500

# ---------------------- ANALIZA: PARAMETRY ----------------------

# === FUNKCJA INTERPRETACJI ===
# dodałam 7.12 – musi być nad analiza_parametrow
def interpretuj(value, opis):
    """Prosta interpretacja wyniku w %."""
    percent = value * 100

    if percent >= 85:
        return f"🌟 Bardzo dobra {opis} (wynik: {percent:.1f}%)"
    elif percent >= 60:
        return f"👍 Dobra {opis}, ale można ją poprawić (wynik: {percent:.1f}%)"
    elif percent >= 30:
        return f"⚠️ Słaba {opis} – wymaga pracy (wynik: {percent:.1f}%)"
    else:
        return f"❌ Bardzo niska {opis} (wynik: {percent:.1f}%)"


# === (NOWA FUNKCJA) WYKRES INTONACJI ===
# zmieniłam 7.12 – poprawiono pyin (3 wartości)
def rysuj_wykres_intonacji(ref_path, user_path, output_file="intonation_plot.png"):
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Wczytanie plików
    y_ref, sr_ref = librosa.load(ref_path, sr=None)
    y_usr, sr_usr = librosa.load(user_path, sr=None)

    # PYIN zwraca 3 wartości — poprawione!
    pitch_ref, _, _ = librosa.pyin(y_ref, fmin=50, fmax=2000, sr=sr_ref)   # zmieniłam 7.12
    pitch_usr, _, _ = librosa.pyin(y_usr, fmin=50, fmax=2000, sr=sr_usr)   # zmieniłam 7.12

    # Czyszczenie NaN
    def clean(p):
        p = np.array(p)
        idx = np.where(~np.isnan(p))[0]
        if len(idx) < 2:
            return np.zeros_like(p)
        return np.interp(np.arange(len(p)), idx, p[idx])

    pitch_ref = clean(pitch_ref)
    pitch_usr = clean(pitch_usr)

    # Dopasowanie długości
    L = min(len(pitch_ref), len(pitch_usr))
    pitch_ref = pitch_ref[:L]
    pitch_usr = pitch_usr[:L]
    time = np.linspace(0, L / sr_ref, L)

    # Rysowanie
    plt.figure(figsize=(10, 3.2))
    plt.plot(time, pitch_ref, label="Reference", color="gold")
    plt.plot(time, pitch_usr, label="User", color="cyan")
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch [Hz]")
    plt.title("Intonation – pitch contour comparison")
    plt.grid(alpha=0.3)
    plt.legend()

    save_path = os.path.join("static", output_file)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()

    return output_file


# === ROUTE ANALIZY PARAMETRÓW ===
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
        wynik, interpretacje = analiza_parametrow(ref_path, user_path)

        # 1) Histogram
        wykres_file = rysuj_histogram_parametrow(wynik)

        # 2) Intonacja (jedyne co generujesz dynamicznie)
        wykres_intonation = rysuj_wykres_intonacji(ref_path, user_path)

        # 3) Pozostałe wykresy – pliki JUŻ istnieją w static/
        wykres_rytm = "rhythm_plot.png"
        wykres_vibrato = "vibrato_plot.png"
        wykres_volume = "volume_plot.png"
        wykres_vq = "voice_quality_plot.png"
        wykres_pron = "pronunciation_plot.png"
        wykres_range = "pitch_range_plot.png"

        # 4) Zwracamy WSZYSTKO do HTML
        return render_template(
            "parametry.html",
            analiza=wynik,
            interpretacje=interpretacje,
            wykres_path=wykres_file,
            wykres_intonation_path=wykres_intonation,
            wykres_rytm_path=wykres_rytm,
            wykres_vibrato_path=wykres_vibrato,
            wykres_volume_path=wykres_volume,
            wykres_vq_path=wykres_vq,
            wykres_pron_path=wykres_pron,
            wykres_range_path=wykres_range
        )

    except Exception as e:
        return f"Błąd analizy: {str(e)}", 500


# === GŁÓWNA FUNKCJA ANALIZY PARAMETRÓW ATSIP ===
def analiza_parametrow(ref_path, user_path):
    import numpy as np
    import librosa

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

        vq_ref = float(np.sum(harm_ref) /
                       (np.sum(harm_ref) + np.sum(per_ref) + 1e-10))
        vq_usr = float(np.sum(harm_usr) /
                       (np.sum(harm_usr) + np.sum(per_usr) + 1e-10))
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
        Intonation, Rhythm, Vibrato, Volume,
        VoiceQuality, Pronunciation, PitchDynamicRange
    ]) * 100.0

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

    interpretacje = {
        "🎵 Intonation": interpretuj(Intonation, "intonacja — trafność wysokości dźwięków"),
        "⏱️ Rhythm": interpretuj(Rhythm, "rytm — zgodność tempa z wzorcem"),
        "🌊 Vibrato": interpretuj(Vibrato, "vibrato — naturalne drganie wysokości"),
        "🔊 Volume": interpretuj(Volume, "głośność — poziom dźwięku względem wzorca"),
        "🧬 Voice Quality": interpretuj(VoiceQuality, "barwa i czystość dźwięku"),
        "🗣️ Pronunciation": interpretuj(Pronunciation, "wymowa i artykulacja"),
        "🔁 Pitch Dynamic Range": interpretuj(PitchDynamicRange, "zakres dynamiczny tonu")
    }

    return wynik, interpretacje


# === HISTOGRAM PARAMETRÓW ATSIP ===
def rysuj_histogram_parametrow(wyniki, output_file="parametry_histogram.png"):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import os

    param_keys = [
        "🎵 Intonation", "⏱️ Rhythm", "🌊 Vibrato",
        "🔊 Volume", "🧬 Voice Quality",
        "🗣️ Pronunciation", "🔁 Pitch Dynamic Range"
    ]

    labels = [k.split(" ", 1)[-1] for k in param_keys]
    values = [float(wyniki[k].replace(" %", "")) for k in param_keys]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(labels, values, color="0.2", edgecolor="black", width=0.6)

    plt.ylim(0, 100)
    plt.ylabel("Ocena [%]", fontsize=11)
    plt.title("Performance of individual perceptual parameters (ATSIP 2018 model)",
              fontsize=12, pad=15)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val / 2,
            f"{val:.0f}%", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    save_path = os.path.join("static", output_file)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file

# ---------------------- ANALIZA: PARAMETRY 2 ----------------------
@app.route("/analizuj_parametry2", methods=["POST"])
def analizuj_parametry2():
    if "audio" not in request.files:
        return "Brak pliku audio", 400

    # Pobieramy częstotliwość z formularza
    try:
        target_freq = float(request.form.get("target_freq", 0))
    except:
        return "Błąd: brak częstotliwości docelowej!", 400

    audio_file = request.files["audio"]
    filename = audio_file.filename.lower()

    os.makedirs(RECORD_DIR, exist_ok=True)

    webm_path = os.path.join(RECORD_DIR, "parametry2_input.webm")
    wav_path = os.path.join(RECORD_DIR, "parametry2_input.wav")

    # Jeśli wgrywany jest WAV → nie konwertujemy
    if filename.endswith(".wav"):
        audio_file.save(wav_path)

    # Jeśli WEBM → konwersja jak wcześniej
    elif filename.endswith(".webm"):
        audio_file.save(webm_path)
        AudioSegment.from_file(webm_path, format="webm").export(wav_path, format="wav")

    else:
        return "Obsługiwane formaty: .wav, .webm", 400

    try:
        analiza, feedback, wykres_path = analiza_czystosci(wav_path, target_freq)
        return render_template("parametry2.html",
                               analiza=analiza,
                               feedback=feedback,
                               wykres_path=wykres_path)
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
