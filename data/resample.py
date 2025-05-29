import os
import librosa
import soundfile as sf

root_folder = "audio_folder"

# Count total number of files to process
total_files = sum(len(files) for _, _, files in os.walk(root_folder) if any(f.endswith(".wav") for f in files))

processed_files = 0

for subdir, _, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".wav"):  # Change if using a different format
            file_path = os.path.join(subdir, file)
            y, sr = librosa.load(file_path, sr=44100)  # Load with original SR
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sf.write(file_path, y_resampled, 22050)  # Overwrite with resampled audio
            
            processed_files += 1
            print(f"Processed {processed_files}/{total_files} files", end="\r")

print("\nResampling complete!")
