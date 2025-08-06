import os
import json

# Change this to your directory
directory =  r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"

# List to store names of matching files
matching_files = []

# Loop over all files in the directory
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Navigate to fair-assessment -> howfairis_score
            howfairis_score = data.get("fair-assessment", {}).get("howfairis_score", 0)

            # Check if score > 0
            if howfairis_score > 0:
                matching_files.append(filename)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

print(f"\nTotal count: {len(matching_files)}")

