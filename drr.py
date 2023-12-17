import os
import zipfile
import tarfile
from fuzzywuzzy import fuzz
from collections import defaultdict
import threading
import concurrent.futures
import hashlib
import pandas as pd
import mimetypes  # Added library for MIME type

def fuzzy_match(file_name1, file_name2, threshold):
    return fuzz.ratio(file_name1, file_name2) > threshold

def get_file_info(file_path):
    _, extension = os.path.splitext(file_path)
    file_size = os.path.getsize(file_path)
    mime_type, mime_encoding = mimetypes.guess_type(file_path)
    return extension.lower(), file_size, mime_type, mime_encoding

def is_considered_file(file_path):
    return os.path.isfile(file_path)

def convert_size(size_in_bytes):
    if size_in_bytes > 1000000:
        return f"{size_in_bytes / 1000000:.2f} MB"
    elif size_in_bytes > 1000:
        return f"{size_in_bytes / 1000:.2f} KB"
    else:
        return f"{size_in_bytes} bytes"

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
    unique_file_types = defaultdict(lambda: {'types': set(), 'size': 0})
    all_file_info = {}
    exact_duplicates = defaultdict(set)
    fuzzy_duplicates = defaultdict(set)
    metadata_info = []  # List to store metadata information
    lock = threading.Lock()

    def process_files(folder_name, file_paths):
        local_unique_file_types = {}
        for file_path in file_paths:
            if is_considered_file(file_path):
                file_type, file_size, mime_type, mime_encoding = get_file_info(file_path)
                if file_path not in local_unique_file_types:
                    local_unique_file_types[file_path] = {'types': set(), 'size': 0}
                local_unique_file_types[file_path]['types'].add(file_type)
                local_unique_file_types[file_path]['size'] += file_size

                # Store file info for later duplicate check
                all_file_info[file_path] = {'types': set(), 'size': 0}
                all_file_info[file_path]['types'].add(file_type)
                all_file_info[file_path]['size'] += file_size

                # Collect metadata information
                metadata_info.append({
                    'File': file_path,
                    'File Name': os.path.basename(file_path),
                    'File Type': file_type,
                    'File Size': convert_size(file_size),
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

    def explore_folders_in_path(folder_path):
        nonlocal unique_file_types
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                if file.endswith(('.zip', '.tar', '.gz', '.bz2', '.rar', '.7z')):
                    print(f"Exploring contents of: {file_path}")
                    extract_folder = os.path.join(root, os.path.splitext(file)[0])
                    extract_archive(file_path, extract_folder)
                    process_folder(extract_folder)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(explore_folders_in_path, base_path),
            executor.submit(process_folder, base_path)
        ]

        # Wait for threads to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Error: {exc}")

        # Ensure all threads are finished before proceeding
        executor.shutdown(wait=True)

    # Find common files using fuzzy matching for file name and size
    processed_files = set()  # To keep track of processed files
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
                # Exact match
                current_exact_duplicates.add(file_path2)
                processed_files.add(file_path2)
            elif (
                file_path1 != file_path2 and
                fuzzy_match(os.path.basename(file_path1), os.path.basename(file_path2), 80) and
                info1['size'] == info2['size']
            ):
                # Slightly similar match
                current_fuzzy_duplicates.add(file_path2)
                processed_files.add(file_path2)

        if current_exact_duplicates:
            exact_duplicates[file_path1] = current_exact_duplicates
        if current_fuzzy_duplicates:
            fuzzy_duplicates[file_path1] = current_fuzzy_duplicates

    # Create separate DataFrames for exactly the same and slightly similar files
    columns = ['File1', 'File1Path', 'File2', 'File2Path', 'ContentMatch']
    df_exact = pd.DataFrame(columns=columns)
    df_fuzzy = pd.DataFrame(columns=columns)

    # Populate DataFrames
    for file_path1, duplicate_addresses in exact_duplicates.items():
        hash1 = hash_file(file_path1)
        for file_path2 in duplicate_addresses:
            hash2 = hash_file(file_path2)
            content_match = hash1 == hash2
            df_exact = pd.concat([df_exact, pd.DataFrame({
                'File1': [os.path.basename(file_path1)],
                'File1Path': [file_path1],
                'File2': [os.path.basename(file_path2)],
                'File2Path': [file_path2],
                'ContentMatch': [content_match],
            })], ignore_index=True)

    for file_path1, duplicate_addresses in fuzzy_duplicates.items():
        hash1 = hash_file(file_path1)
        for file_path2 in duplicate_addresses:
            hash2 = hash_file(file_path2)
            content_match = hash1 == hash2
            df_fuzzy = pd.concat([df_fuzzy, pd.DataFrame({
                'File1': [os.path.basename(file_path1)],
                'File1Path': [file_path1],
                'File2': [os.path.basename(file_path2)],
                'File2Path': [file_path2],
                'ContentMatch': [content_match],
            })], ignore_index=True)

    # Print metadata information
    metadata_df = pd.DataFrame(metadata_info)
    metadata_excel_filename = "metadata.xlsx"
    metadata_df.to_excel(metadata_excel_filename, index=False)
    print(f"\nMetadata DataFrame saved to {metadata_excel_filename}")

    # Print exactly duplicated files
    print("\nExactly Same Files:")
    if not df_exact.empty:
        print(df_exact)
    else:
        print("No exactly the same files found.")

    # Print slightly similar files
    print("\nSlightly Similar Files:")
    if not df_fuzzy.empty:
        print(df_fuzzy)
    else:
        print("No slightly similar files found.")

    # Save the DataFrames to Excel files
    excel_exact_filename = "exact_duplicates.xlsx"
    excel_fuzzy_filename = "fuzzy_duplicates.xlsx"
    siamese_dataset_excel_filename = "siamese_dataset.xlsx"

    df_exact.to_excel(excel_exact_filename, index=False)
    df_fuzzy.to_excel(excel_fuzzy_filename, index=False)

    # Create Siamese Dataset
    siamese_dataset = pd.DataFrame(columns=['File1Name', 'File1', 'File2Name', 'File2', 'HashFile1', 'HashFile2', 'Match'])

    for file_path1, duplicate_addresses in exact_duplicates.items():
        hash1 = hash_file(file_path1)
        for file_path2 in duplicate_addresses:
            hash2 = hash_file(file_path2)
            content_match = hash1 == hash2
            siamese_dataset = pd.concat([siamese_dataset, pd.DataFrame({
                'File1Name': [os.path.basename(file_path1)],
                'File2Name': [os.path.basename(file_path2)],
                'HashFile1': [hash1],
                'HashFile2': [hash2],
                'Match': [int(content_match)],
            })], ignore_index=True)

    for file_path1, duplicate_addresses in fuzzy_duplicates.items():
        hash1 = hash_file(file_path1)
        for file_path2 in duplicate_addresses:
            hash2 = hash_file(file_path2)
            content_match = hash1 == hash2
            siamese_dataset = pd.concat([siamese_dataset, pd.DataFrame({
                'File1Name': [os.path.basename(file_path1)],
                'File2Name': [os.path.basename(file_path2)],
                'HashFile1': [hash1],
                'HashFile2': [hash2],
                'Match': [int(content_match)],
            })], ignore_index=True)

    siamese_dataset.to_excel(siamese_dataset_excel_filename, index=False)

    print(f"\nExactly Same Files DataFrame saved to {excel_exact_filename}")
    print(f"\nSlightly Similar Files DataFrame saved to {excel_fuzzy_filename}")
    print(f"\nSiamese Dataset saved to {siamese_dataset_excel_filename}")

if __name__ == "__main__":
    base_path = r"D:\\adhvik\\adh\\Hackathon\\space hack\\Data RR\\dataset1"
    explore_and_find_duplicates(base_path)
