import h5py
import numpy as np
import pickle
import os

# Path to the root directory containing the subject folders (e.g., oliver, chemistry)
base_dir = "/datasets/PATS/pats/data/processed"  # Adjust path to where the segregated files are
output_file = "filepath/extracted_text.txt"  # Output file path

# Open the output file in write mode
with open(output_file, "w") as output:

    # Iterate over each subject folder (e.g., oliver, chemistry)
    for subject in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject)
        
        if not os.path.isdir(subject_path):  # Skip if it's not a directory
            continue

        # Iterate over each file in the subject folder
        for file in os.listdir(subject_path):
            if file.endswith(".h5"):
                file_path = os.path.join(subject_path, file)
                
                # Extract the base filename (without folder prefix) to match the .npy file name
                base_filename = os.path.splitext(file)[0]  # Remove the .h5 extension
                
                # Construct the corresponding .npy filename
                npy_filename = f"{subject}#{base_filename}.npy"  # Construct the corresponding .npy filename

                # Check if there's a corresponding .npy file in the segregated feature folder
                npy_dir = "/hdd/lokesh/models/datasets/PATS/pats/data/pose_features1"  # Path to the folder where .npy files are moved
                npy_file_path = os.path.join(npy_dir, subject, npy_filename)

                # Proceed if the corresponding .npy file exists
                if os.path.exists(npy_file_path):
                    # Open the HDF5 file
                    with h5py.File(file_path, "r") as f:
                        text_key = "text/meta/block0_values"  # The text key you are extracting
                        
                        if text_key in f:
                            text_data = f[text_key][()]  # Load the dataset

                            # Case 1: If the text is stored as a serialized object, unpickle it
                            if text_data.dtype.kind == "O":
                                extracted_text = pickle.loads(text_data[0])  # Deserialize
                            # Case 2: If stored as bytes, decode each entry
                            elif isinstance(text_data[0], bytes):
                                extracted_text = [t.decode("utf-8") for t in text_data]
                            # Case 3: If stored as NumPy arrays, flatten and convert to strings
                            else:
                                extracted_text = np.array(text_data).flatten()  # Flatten the array

                                # Convert each element to string if it's a NumPy ndarray
                                extracted_text = [
                                    str(word.item()) if isinstance(word, np.ndarray) else str(word)
                                    for word in extracted_text
                                ]
                            
                            # Flatten any remaining nested structure and convert to string
                            flat_text = []
                            for item in extracted_text:
                                if isinstance(item, np.ndarray):
                                    flat_text.extend(item.tolist())  # Flatten if still a NumPy array
                                elif isinstance(item, list):
                                    flat_text.extend(item)  # Flatten if it's a list
                                else:
                                    flat_text.append(str(item))  # Append as a string

                            # Now join all words into a sentence
                            sentence = " ".join(flat_text)

                            # Fix spacing & punctuation issues
                            sentence = sentence.replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" n't", "n't")

                            # Format the output line as required
                            output_line = f"{subject}/{subject}#{base_filename}|{sentence}|{sentence}\n"

                            # Write the output to the file
                            output.write(output_line)

                    print(f"Processed: {file_path} and found corresponding .npy file.")

print("✅ Extraction and saving completed.")
