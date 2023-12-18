import os
import zipfile
import tarfile
from fuzzywuzzy import fuzz
from collections import defaultdict
import concurrent.futures
import hashlib
import pandas as pd
import mimetypes
import threading  # Import threading module

# Removed the download_file function

def fuzzy_match(file_name1, file_name2, threshold):
    return fuzz.ratio(file_name1, file_name2) > threshold

def get_file_info(file_path):
    _, extension = os.path.splitext(file_path)
    file_size = os.path.getsize(file_path)
    mime_type, mime_encoding = mimetypes.guess_type(file_path)
    return extension.lower(), file_size, mime_type, mime_encoding

def is_considered_file(file_path):
    return os.path.isfile(file_path)

# Removed the convert_size function

def extract_archive(archive_path, extract_folder):
    _, extension = os.path.splitext(archive_path)

    if extension.lower() == '.tar':
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(path=extract_folder)
    elif extension.lower() == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

def hash_file(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def explore_and_find_duplicates(base_path):
    # Removed the local_base_path directory creation
    # Using the base_path directly for local host

    unique_file_types = defaultdict(lambda: {'types': set(), 'size': 0})
    all_file_info = {}
    exact_duplicates = defaultdict(set)
    fuzzy_duplicates = defaultdict(set)
    metadata_info = []
    lock = threading.Lock()  # Use threading.Lock() for thread synchronization

    def process_files(folder_name, file_paths):
        local_unique_file_types = {}
        for file_path in file_paths:
            if is_considered_file(file_path):
                file_type, file_size, mime_type, mime_encoding = get_file_info(file_path)
                if file_path not in local_unique_file_types:
                    local_unique_file_types[file_path] = {'types': set(), 'size': 0}
                local_unique_file_types[file_path]['types'].add(file_type)
                local_unique_file_types[file_path]['size'] += file_size

                all_file_info[file_path] = {'types': set(), 'size': 0}
                all_file_info[file_path]['types'].add(file_type)
                all_file_info[file_path]['size'] += file_size

                metadata_info.append({
                    'File': file_path,
                    'File Name': os.path.basename(file_path),
                    'File Type': file_type,
                    'File Size': file_size,
                    'MIME Type': mime_type,
                    'MIME Encoding': mime_encoding
                })

        with lock:
            for file_path, file_info in local_unique_file_types.items():
                if file_path not in unique_file_types:
                    unique_file_types[file_path] = {'types': set(), 'size': 0}
                unique_file_types[file_path]['types'].update(file_info['types'])
                unique_file_types[file_path]['size'] += file_info['size']

    def process_folder(folder_path):
        nonlocal unique_file_types
        for root, _, files in os.walk(folder_path):
            file_paths = [os.path.join(root, file) for file in files]
            process_files(root, file_paths)

    def explore_compressed_folder(folder_path):
        nonlocal unique_file_types
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if file.endswith(('.zip', '.tar', '.gz', '.bz2', '.rar', '.7z')):
                print(f"Exploring contents of: {file_path}")
                extract_folder = os.path.join(folder_path, os.path.splitext(file)[0])
                extract_archive(file_path, extract_folder)
                process_folder(extract_folder)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(explore_compressed_folder, base_path),
            executor.submit(process_folder, base_path)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Error: {exc}")

    processed_files = set()
    for file_path1, info1 in all_file_info.items():
        if file_path1 in processed_files:
            continue
        current_exact_duplicates = set()
        current_fuzzy_duplicates = set()
        for file_path2, info2 in all_file_info.items():
            if (
                file_path1 != file_path2 and
                fuzzy_match(os.path.basename(file_path1), os.path.basename(file_path2), 99) and
                info1['size'] == info2['size']
            ):
                current_exact_duplicates.add(file_path2)
                processed_files.add(file_path2)
            elif (
                file_path1 != file_path2 and
                fuzzy_match(os.path.basename(file_path1), os.path.basename(file_path2), 80) and
                info1['size'] == info2['size']
            ):
                current_fuzzy_duplicates.add(file_path2)
                processed_files.add(file_path2)

        if current_exact_duplicates:
            exact_duplicates[file_path1] = current_exact_duplicates
        if current_fuzzy_duplicates:
            fuzzy_duplicates[file_path1] = current_fuzzy_duplicates

    columns = ['File1', 'File2', 'ContentMatch', 'File1Name', 'File2Name']
    df_exact = pd.DataFrame(columns=columns)
    df_fuzzy = pd.DataFrame(columns=columns)

    for file_path1, duplicate_addresses in exact_duplicates.items():
        for file_path2 in duplicate_addresses:
            hash1 = hash_file(file_path1)
            hash2 = hash_file(file_path2)
            content_match = hash1 == hash2
            df_exact = pd.concat([df_exact, pd.DataFrame({
                'File1': [file_path1] * len(duplicate_addresses),
                'File2': list(duplicate_addresses),
                'ContentMatch': [content_match] * len(duplicate_addresses),
                'File1Name': [os.path.basename(file_path1)] * len(duplicate_addresses),
                'File2Name': [os.path.basename(file_path2) for file_path2 in duplicate_addresses],
            })], ignore_index=True)

    for file_path1, duplicate_addresses in fuzzy_duplicates.items():
        for file_path2 in duplicate_addresses:
            hash1 = hash_file(file_path1)
            hash2 = hash_file(file_path2)
            content_match = hash1 == hash2
            df_fuzzy = pd.concat([df_fuzzy, pd.DataFrame({
                'File1Name': [os.path.basename(file_path1)],
                'File1': [file_path1],
                'File2Name': [os.path.basename(file_path2)],
                'File2': [file_path2],
                'ContentMatch': [content_match],
            })], ignore_index=True)

    metadata_df = pd.DataFrame(metadata_info)
    metadata_excel_filename = "metadata.xlsx"
    metadata_df.to_excel(metadata_excel_filename, index=False)
    print(f"\nMetadata DataFrame saved to {metadata_excel_filename}")

    print("\nExactly Same Files:")
    if not df_exact.empty:
        print(df_exact)
    else:
        print("No exactly same files found.")

    print("\nSlightly Similar Files:")
    if not df_fuzzy.empty:
        print(df_fuzzy)
    else:
        print("No slightly similar files found.")

    excel_exact_filename = "exact_duplicates.xlsx"
    excel_fuzzy_filename = "fuzzy_duplicates.xlsx"

    df_exact.to_excel(excel_exact_filename, index=False)
    df_fuzzy.to_excel(excel_fuzzy_filename, index=False)

    print(f"\nExactly Same Files DataFrame saved to {excel_exact_filename}")
    print(f"\nSlightly Similar Files DataFrame saved to {excel_fuzzy_filename}")

if __name__ == "__main__":
    base_path = "http://localhost:8000/"  # Replace with the path to your local files
    explore_and_find_duplicates(base_path)