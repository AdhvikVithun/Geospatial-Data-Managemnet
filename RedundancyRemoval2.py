import os
import zipfile
import tarfile
from fuzzywuzzy import fuzz
from collections import defaultdict
import threading
import concurrent.futures
import hashlib
import pandas as pd
import mimetypes
import time
import multiprocessing
import matplotlib.pyplot as plt

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
    elif extension.lower() == '.gz':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

def hash_file(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_fuzzy_duplicates(file_info_pair):
    fuzzy_duplicates = set()
    file_path1, info1, file_path2, info2 = file_info_pair

    if (
        file_path1 != file_path2 and
        fuzzy_match(os.path.basename(file_path1), os.path.basename(file_path2), 99) and
        info1['size'] == info2['size']
    ):
        fuzzy_duplicates.add((file_path1, file_path2))

    return fuzzy_duplicates

def process_fuzzy_duplicates(all_file_info):
    processed_files = set()
    fuzzy_duplicates = defaultdict(set)

    file_info_pairs = [(file_path1, info1, file_path2, info2)
                       for file_path1, info1 in all_file_info.items()
                       for file_path2, info2 in all_file_info.items()
                       if file_path1 not in processed_files]

    # Use maximum number of threads
    num_threads = min(multiprocessing.cpu_count() * 2, 32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Execute the fuzzy matching in parallel
        fuzzy_results = list(executor.map(find_fuzzy_duplicates, file_info_pairs))

    # Merge the results from different threads
    for result_set in fuzzy_results:
        for file_pair in result_set:
            fuzzy_duplicates[file_pair[0]].add(file_pair[1])
            fuzzy_duplicates[file_pair[1]].add(file_pair[0])

    return dict(fuzzy_duplicates)


def explore_and_find_duplicates(base_path):
    folder_times = defaultdict(float)  # To store time taken for each folder
    unique_file_types = defaultdict(lambda: {'types': set(), 'size': 0})
    all_file_info = {}
    exact_duplicates = defaultdict(set)
    metadata_info = set()  # Use a set to store unique metadata information
    lock = threading.Lock()

    def process_files(folder_name, file_paths):
        local_unique_file_types = {}
        local_metadata_info = set()  # Store metadata information for this run
        for file_path in file_paths:
            if is_considered_file(file_path):
                print(f"Processing file: {file_path}")
                try:
                    start_time = time.time()  # Record start time for each file
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
                    local_metadata_info.add((file_path, os.path.basename(file_path), file_type, convert_size(file_size), mime_type, mime_encoding))

                    end_time = time.time()  # Record end time for each file
                    elapsed_time = end_time - start_time
                    folder_times[folder_name] += elapsed_time

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        with lock:
            for file_path, file_info in local_unique_file_types.items():
                if file_path not in unique_file_types:
                    unique_file_types[file_path] = {'types': set(), 'size': 0}
                unique_file_types[file_path]['types'].update(file_info['types'])
                unique_file_types[file_path]['size'] += file_info['size']

        # Update the global metadata_info with local_metadata_info
        with lock:
            metadata_info.update(local_metadata_info)

    def process_folder(folder_path):
        nonlocal unique_file_types
        start_time = time.time()  # Record start time for each folder
        try:
            for root, _, files in os.walk(folder_path):
                file_paths = [os.path.join(root, file) for file in files]
                print(f"Processing folder: {root}")
                process_files(os.path.basename(root), file_paths)
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")

        end_time = time.time()  # Record end time for each folder
        elapsed_time = end_time - start_time
        folder_times[os.path.basename(folder_path)] = elapsed_time

    def explore_compressed_folder(folder_path):
        nonlocal unique_file_types
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if file.endswith(('.zip', '.tar', '.gz', '.bz2', '.rar', '.7z')):
                print(f"Exploring contents of: {file_path}")
                extract_folder = os.path.join(folder_path, os.path.splitext(file)[0])
                extract_archive(file_path, extract_folder)
                # Now, process the contents of the extracted folder
                process_folder(extract_folder)
            else:
                # If it's not a compressed file, process it directly
                process_folder(file_path)

    # Use maximum number of threads
    num_threads = min(multiprocessing.cpu_count() * 2, 32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(explore_compressed_folder, base_path),
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

    # Parallelize hashing process using maximum number of processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        hash_values = list(executor.map(hash_file, all_file_info.keys()))
        all_file_info_hashes = dict(zip(all_file_info.keys(), hash_values))
    
    # Find common files using fuzzy matching for file name and size
    fuzzy_duplicates = process_fuzzy_duplicates(all_file_info)

    # Create separate DataFrames for exactly same and slightly similar files
    columns = ['File1', 'File2', 'ContentMatch', 'File1Name', 'File2Name']
    df_exact = pd.DataFrame(columns=columns)
    df_fuzzy = pd.DataFrame(columns=columns)

    # Populate DataFrames
    for file_path1, duplicate_addresses in exact_duplicates.items():
        for file_path2 in duplicate_addresses:
            hash1 = all_file_info_hashes[file_path1]
            hash2 = all_file_info_hashes[file_path2]
            content_match = hash1 == hash2
            df_exact = pd.concat([df_exact, pd.DataFrame({
                'File1': [file_path1] * len(duplicate_addresses),
                'File2': [file_path2] * len(duplicate_addresses),
                'ContentMatch': [content_match] * len(duplicate_addresses),
                'File1Name': [os.path.basename(file_path1)] * len(duplicate_addresses),
                'File2Name': [os.path.basename(file_path2)] * len(duplicate_addresses),
            })], ignore_index=True)

    for file_path1, duplicate_addresses in fuzzy_duplicates.items():
        for file_path2 in duplicate_addresses:
            hash1 = all_file_info_hashes[file_path1]
            hash2 = all_file_info_hashes[file_path2]
            content_match = hash1 == hash2
            df_fuzzy = pd.concat([df_fuzzy, pd.DataFrame({
                'File1Name': [os.path.basename(file_path1)],
                'File1': [file_path1],
                'File2Name': [os.path.basename(file_path2)],
                'File2': [file_path2],
                'ContentMatch': [content_match],
            })], ignore_index=True)

    # Print metadata information
    metadata_df = pd.DataFrame(metadata_info, columns=['File', 'File Name', 'File Type', 'File Size', 'MIME Type', 'MIME Encoding'])
    metadata_excel_filename = "metadata.xlsx"
    metadata_df.to_excel(metadata_excel_filename, index=False)
    print(f"\nMetadata DataFrame saved to {metadata_excel_filename}")

    # Print exactly duplicated files
    print("\nExactly Same Files:")
    if not df_exact.empty:
        print(df_exact)
    else:
        print("No exactly same files found.")

    # Print slightly similar files
    print("\nSlightly Similar Files:")
    if not df_fuzzy.empty:
        print(df_fuzzy)
    else:
        print("No slightly similar files found.")

    # Save the DataFrames to Excel files
    excel_exact_filename = "exact_duplicates.xlsx"
    excel_fuzzy_filename = "fuzzy_duplicates.xlsx"

    df_exact.to_excel(excel_exact_filename, index=False)
    df_fuzzy.to_excel(excel_fuzzy_filename, index=False)

    print(f"\nExactly Same Files DataFrame saved to {excel_exact_filename}")
    print(f"\nSlightly Similar Files DataFrame saved to {excel_fuzzy_filename}")

    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2

    # Print the overall elapsed time
    print(f"\nTotal execution time: {elapsed_time2} seconds")

if __name__ == "__main__":
    base_path = r"D:\adhvik\adh\Hackathon\space hack\finals data"
    #D:\adhvik\adh\Hackathon\space hack\Data RR\data Set\topic12\dataset1
    #D:\adhvik\adh\Hackathon\space hack\zip
    #D:\adhvik\adh\Hackathon\space hack\finals data\Topic-1\Geospatial Data Mangement
    #D:\adhvik\adh\Hackathon\space hack\siamese data lu lc\Sen-2 LULC\train_images\train
    # Start the timer
    start_time2 = time.time()
    print("Timer started")
    explore_and_find_duplicates(base_path)
