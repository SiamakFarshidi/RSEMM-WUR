#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    # === Make sure each string here matches exactly what you see in `dir *.py` ===

   # "1_zenodoProcessing.py", 
   # "2_searchTermExtension.py", 
   # "3_itetarive-se-practises-evaluation.py",
   # "4_itetarive-ai-ml-ops-evaluation.py",
   # "5_itetarive-fairness-assessment.py",
    scripts = [

        "6_itetarive-code-gen-detection.py"
    ]

    for script_name in scripts:
        # Build a full path to the script in the current working directory
        script_path = os.path.join(os.getcwd(), script_name)

        if not os.path.isfile(script_path):
            print(f'Error: "{script_name}" does not exist in the current directory.')
            sys.exit(1)

        print(f"\n→ Running {script_name} …")
        result = subprocess.run([sys.executable, script_path])

        if result.returncode != 0:
            print(f'✖ Script "{script_name}" exited with code {result.returncode}. Aborting batch.')
            sys.exit(result.returncode)

    print("\n✓ All scripts completed successfully.")

if __name__ == "__main__":
    main()
