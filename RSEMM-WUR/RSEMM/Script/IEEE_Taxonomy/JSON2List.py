#!/usr/bin/env python3
"""
JSON2List.py

Reads a JSON file containing a hierarchical structure of nodes,
each with a "name" field (and possibly a "children" list). It
extracts all unique names and writes them to a text file, one per
line, sorted alphabetically.

Usage: (hardcoded filenames)
    python JSON2List.py
"""

import json
import sys
from pathlib import Path

def collect_names(nodes, names_set):
    """
    Recursively traverse a list of nodes, adding each node's "name"
    (if present) into names_set. If a node has a "children" list,
    recurse into it.
    """
    for node in nodes:
        name = node.get("name")
        if name:
            names_set.add(name)
        children = node.get("children", [])
        if isinstance(children, list) and children:
            collect_names(children, names_set)

def main():
    # Wrap these strings in Path(...) so we can call .is_file() and .open()
    input_path = Path("ieee_taxonomy_clean.json")
    output_path = Path("ieee_taxonomy_list.txt")

    # Check that input file exists
    if not input_path.is_file():
        print(f"Error: {input_path} does not exist or is not a file.")
        sys.exit(1)

    # Load JSON data
    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    # Collect unique names
    unique_names = set()
    if isinstance(data, list):
        collect_names(data, unique_names)
    else:
        # If the top‐level JSON is a dict, wrap it in a list
        collect_names([data], unique_names)

    # Sort names alphabetically
    sorted_names = sorted(unique_names)

    # Write to output file
    try:
        with output_path.open("w", encoding="utf-8") as f_out:
            for name in sorted_names:
                f_out.write(f"{name}\n")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")
        sys.exit(1)

    print(f"Extracted {len(sorted_names)} unique names and wrote them to {output_path}")

if __name__ == "__main__":
    main()
