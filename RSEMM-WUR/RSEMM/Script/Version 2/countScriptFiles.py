import os
import json

# Change this to your directory
directory = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"

# Initialize sum
total_selected_recent_files = 0

# Loop over all files in directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Navigate to code-gen-evaluation -> #Selected Recent Files
            selected_recent_files = data.get("repo-profile", {}) \
                                        .get("code-gen-evaluation", {}) \
                                        .get("#Selected Recent Files", 0)
            
            # Add to total
            total_selected_recent_files += selected_recent_files

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Print the total sum
print(f"Total #Selected Recent Files across all JSON files: {total_selected_recent_files}")

