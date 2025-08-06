import os
import json
import hashlib

def get_file_hash(filepath):
    """Compute a hash of the file's contents."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_directory(root_dir):
    unique_files = {}
    duplicates = []  # list to hold paths to duplicate files
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            doi_url = data.get("doi_url")
            if not doi_url:
                print(f"File {filepath} does not contain a doi_url; skipping.")
                continue

            file_size = os.path.getsize(filepath)
            file_hash = get_file_hash(filepath)

            if doi_url in unique_files:
                stored = unique_files[doi_url]
                if stored["hash"] == file_hash:
                    # The file is identical to what we already have.
                    print(f"Identical duplicate found for {doi_url}: {filepath} is a duplicate of {stored['path']}.")
                    duplicates.append(filepath)
                else:
                    # The file is different; choose the one with more data (larger size).
                    if file_size > stored["size"]:
                        print(f"For {doi_url}, replacing {stored['path']} with larger file {filepath}.")
                        duplicates.append(stored['path'])  # Mark the stored file for removal.
                        unique_files[doi_url] = {"path": filepath, "size": file_size, "hash": file_hash}
                    else:
                        print(f"For {doi_url}, keeping existing file {stored['path']} over {filepath}.")
                        duplicates.append(filepath)
            else:
                unique_files[doi_url] = {"path": filepath, "size": file_size, "hash": file_hash}

    return unique_files, duplicates

def main():
    root_dir = r"C:\Users\User\Downloads\RSE-Analysis\RS"  # Update with your actual directory
    unique_files, duplicates = process_directory(root_dir)
    
    # Remove duplicate files from disk
    for duplicate in duplicates:
        try:
            os.remove(duplicate)
            print(f"Removed duplicate file: {duplicate}")
        except Exception as e:
            print(f"Error removing {duplicate}: {e}")
    
    # Optionally, store the unique doi_urls in a JSON file
    output = {"unique_doi_urls": list(unique_files.keys())}
    output_filename = "unique_doi_urls.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Done. Unique DOI URLs stored in {output_filename}")

if __name__ == '__main__':
    main()
