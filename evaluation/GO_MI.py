import os
import numpy as np
import librosa
from scipy.signal import find_peaks
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# -------- Time binning for MI --------
def bin_times(times, total_duration, num_bins=20):
    bins = np.linspace(0, total_duration, num_bins + 1)
    digitized = np.digitize(times, bins) - 1
    hist, _ = np.histogram(times, bins=bins)
    return hist.reshape(-1, 1)

# -------- CMTD calculation --------
def calculate_cmtd(gesture_file, audio_file):
    gesture_data = np.load(gesture_file)
    gesture_apexes = find_peaks(np.linalg.norm(gesture_data, axis=1))[0]

    audio, sr = librosa.load(audio_file)
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_prominence = np.mean(pitch, axis=0)
    speech_prominence, _ = find_peaks(pitch_prominence, height=np.mean(pitch_prominence))

    total_audio_duration = len(audio) / sr
    gesture_times = gesture_apexes / len(gesture_data) * total_audio_duration
    speech_times = speech_prominence / len(pitch_prominence) * total_audio_duration

    N = min(len(gesture_times), len(speech_times))
    if N == 0:
        return 0.0

    cmtd = np.mean([abs(gesture_times[i] - speech_times[i]) for i in range(N)])
    return cmtd

# -------- MI calculation --------
def calculate_mi(gesture_file, audio_file, num_bins=20):
    gesture_data = np.load(gesture_file)
    gesture_apexes = find_peaks(np.linalg.norm(gesture_data, axis=1))[0]

    audio, sr = librosa.load(audio_file)
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_prominence = np.mean(pitch, axis=0)
    speech_prominence, _ = find_peaks(pitch_prominence, height=np.mean(pitch_prominence))

    total_audio_duration = len(audio) / sr
    gesture_times = gesture_apexes / len(gesture_data) * total_audio_duration
    speech_times = speech_prominence / len(pitch_prominence) * total_audio_duration

    if len(gesture_times) == 0 or len(speech_times) == 0:
        return 0.0

    gesture_hist = bin_times(gesture_times, total_audio_duration, num_bins)
    speech_hist = bin_times(speech_times, total_audio_duration, num_bins)

    mi = mutual_info_regression(gesture_hist, speech_hist.ravel(), random_state=42)
    return mi[0]

# -------- Combined evaluation loop --------
def compute_cmtd_and_mi(gesture_root, tts_audio_root):
    total_cmtd = 0
    total_mi = 0
    count = 0

    for speaker in tqdm(os.listdir(tts_audio_root), desc="Processing Speakers"):
        tts_speaker_path = os.path.join(tts_audio_root, speaker)
        gesture_speaker_path = os.path.join(gesture_root, speaker)

        if not os.path.isdir(tts_speaker_path):
            continue

        for tts_file in tqdm(os.listdir(tts_speaker_path), desc=f"Processing {speaker}", leave=False):
            if not tts_file.endswith(".wav"):
                continue

            base_name = os.path.splitext(tts_file)[0]
            gesture_file = os.path.join(gesture_speaker_path, base_name + ".npy")
            tts_audio_file = os.path.join(tts_speaker_path, tts_file)

            if not (os.path.exists(gesture_file) and os.path.exists(tts_audio_file)):
                continue

            cmtd = calculate_cmtd(gesture_file, tts_audio_file)
            mi = calculate_mi(gesture_file, tts_audio_file)

            total_cmtd += cmtd
            total_mi += mi
            count += 1

            print(f"{speaker}/{tts_file} => CMTD: {cmtd:.4f}s | MI: {mi:.4f}")

    avg_cmtd = total_cmtd / count if count > 0 else 0
    avg_mi = total_mi / count if count > 0 else 0

    print(f"\nAverage CMTD (TTS vs Gestures): {avg_cmtd:.4f}s")
    print(f"Average Mutual Information (TTS vs Gestures): {avg_mi:.4f}")
    return avg_cmtd, avg_mi

# === Example Usage ===
gesture_root = "/PATS/pats/data/pose_features/"
tts_audio_root = "tts_output/"

compute_cmtd_and_mi(gesture_root, tts_audio_root)
