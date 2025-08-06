import os
import json
import pandas as pd

def generate_csv_from_flattened_jsons(root_dir, output_csv="flattened_data.csv"):
    """
    Scan the root_dir (including subdirectories) for JSON files,
    load each JSON (assumed to be flattened), and export a combined CSV.
    """
    data_rows = []
    # Walk the directory structure
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".json"):
                file_path = os.path.join(dirpath, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data_rows.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    if data_rows:
        df = pd.DataFrame(data_rows)
        output_path = os.path.join(root_dir, output_csv)
        df.to_csv(output_path, index=False)
        print(f"CSV file generated at: {output_path}")
    else:
        print("No JSON files found in the directory.")

# Example usage:
if __name__ == "__main__":
    OUTPUT_ROOT = r"C:\Users\User\Downloads\RSE-Analysis\RS_2.0"
    generate_csv_from_flattened_jsons(OUTPUT_ROOT)
