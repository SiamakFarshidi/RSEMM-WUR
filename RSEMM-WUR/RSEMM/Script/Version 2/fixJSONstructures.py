import os
import json

# Path to your JSON files directory
directory =r"C:/Users/User/Source/Repos/SiamakFarshidi/DecisionModelGalaxy/DecisionModelGalaxy/RSEMM/zenodo_records"

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Move "fair-assessment" under "repo-profile"
        fair_assessment = data.pop("fair-assessment", None)

        if fair_assessment:
            if "repo-profile" not in data:
                data["repo-profile"] = {}

            data["repo-profile"]["fair-assessment"] = fair_assessment
        
            # Save updated JSON back to the file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ All JSON files updated successfully.")

