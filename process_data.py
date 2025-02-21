import pandas as pd
import os

# Input and output paths
input_csv = "data/discharge.csv"
output_dir = "discharge_notes"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read CSV in chunks
chunk_size = 100000  # Adjust as needed
subject_files = {}

# Process the CSV file in chunks
for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
    for subject_id, group in chunk.groupby("subject_id"):
        output_file = os.path.join(output_dir, f"{subject_id}.csv")
        
        # Append or create a new file
        if subject_id not in subject_files:
            group.to_csv(output_file, index=False)
            subject_files[subject_id] = True
        else:
            group.to_csv(output_file, mode='a', header=False, index=False)
    print("Processed chunk")

print("Splitting complete. Files are saved in:", output_dir)
