import os
import json
from datetime import datetime

# Set your input and output directories
input_directory = r'C:\Users\User\Downloads\RSE-Analysis\RS3.0'
output_directory = r'C:\Users\User\Downloads\RSE-Analysis\RS4.0'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over every file in the input directory
for filename in os.listdir(input_directory):
    if not filename.endswith('.json'):
        continue  # Skip non-JSON files

    # Open and load the JSON file
    input_filepath = os.path.join(input_directory, filename)
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        try:
            data = json.load(infile)
        except json.JSONDecodeError as e:
            print(f"Error decoding {filename}: {e}")
            continue

    # Initialize counters for the new keys
    gen_ai_coding_files = 0
    code_gen_keyword_total_occurrences = 0

    # Check if "repo_analysis" exists in the JSON data
    if "repo_analysis" in data:
        repo_analysis = data["repo_analysis"]

        # Process "recent_files_analysis" if it exists
        recent_files = repo_analysis.get("recent_files_analysis", [])

        # New: Count total number of analyzed scripts
        analyzed_script_count = len(recent_files)

        for file_analysis in recent_files:
            # Check that "gen-ai" is true; default to False if not present.
            if file_analysis.get("gen-ai", False):
                # Process the last_update date.
                # Remove trailing 'Z' if it exists, to be compatible with datetime.fromisoformat
                last_update_str = file_analysis.get("last_update", "").rstrip("Z")
                try:
                    last_update_date = datetime.fromisoformat(last_update_str)
                except ValueError:
                    # If the date cannot be parsed, skip to the next file
                    print(f"Could not parse date {file_analysis.get('last_update', '')} in {filename}. Skipping entry.")
                    continue

                # Check if the file was updated after 2022 (i.e. year > 2022)
                if last_update_date.year > 2022:
                    gen_ai_coding_files += 1
                    code_gen_keyword_total_occurrences += file_analysis.get("code_gen_keyword_occurrences", 0)

        # Extract additional values from the repo_analysis structure
        average_documentation_ratio = repo_analysis.get("average_documentation_ratio", 0)
        packages_used = repo_analysis.get("packages_used", [])
        readme_length = repo_analysis.get("readme_length", 0)

        # Remove the original "repo_analysis" key to flatten the structure
        del data["repo_analysis"]

        # Add the extracted and computed keys at the root level
        data["average_documentation_ratio"] = average_documentation_ratio
        data["packages_used"] = packages_used
        data["readme_length"] = readme_length
        data["gen-ai-coding-files"] = gen_ai_coding_files
        data["code_gen_keyword_total-occurrences"] = code_gen_keyword_total_occurrences
        data["analyzed-script-count"] = analyzed_script_count

    # Save the modified JSON structure in the output directory
    output_filepath = os.path.join(output_directory, filename)
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Processed and saved: {filename}")

