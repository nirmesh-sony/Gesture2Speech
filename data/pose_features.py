import os
import h5py
import numpy as np

# Define the base directory containing subject folders
base_dir = "/hdd/lokesh/models/datasets/PATS/pats/data/processed/"

# Define the output base directory where extracted .npy files will be saved
output_base_dir = "/hdd/lokesh/models/datasets/PATS/pats/data/pose_features"

# List of subject folders to process
subjects = ["almaram", "angelica"]

# Iterate over each subject folder
for subject in subjects:
    subject_path = os.path.join(base_dir, subject)

    # Ensure the subject folder exists
    if not os.path.isdir(subject_path):
        print(f"Skipping {subject} (folder not found).")
        continue

    # Create the subject-specific output folder
    output_subject_path = os.path.join(output_base_dir, subject)
    os.makedirs(output_subject_path, exist_ok=True)

    # Iterate over all .h5 files in the subject folder
    for filename in os.listdir(subject_path):
        if filename.endswith(".h5"):
            h5_path = os.path.join(subject_path, filename)

            # Extract features
            with h5py.File(h5_path, "r") as f:
                if "pose/normalize" in f:
                    pose_data = np.array(f["pose/normalize"])
                else:
                    print(f"Skipping {filename}, 'pose/normalize' not found.")
                    continue

            # Save the extracted features as .npy in the new output directory
            output_filename = f"{os.path.splitext(filename)[0]}.npy"
            output_path = os.path.join(output_subject_path, output_filename)
            np.save(output_path, pose_data)

            print(f"Saved: {output_path}")

print("Feature extraction completed for all subjects.")
