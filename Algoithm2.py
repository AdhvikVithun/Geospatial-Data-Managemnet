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
import geopandas as gpd

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

def read_geospatial_data(file_path):
    if file_path.lower().endswith(('.shp', '.geojson', '.kml')):
        gdf = gpd.read_file(file_path)
        return gdf

def ingest_geospatial_data(file_path):
    gdf = read_geospatial_data(file_path)
    return gdf

def extract_geospatial_metadata(gdf):
    metadata = {
        'file_type': 'geospatial',
        'geometry_type': gdf.geometry.type.iloc[0],
    }
    return metadata

def classify_geospatial_data(gdf):
    return gdf.geometry.type.iloc[0]

def ingest_non_geospatial_data(file_path):
    # Placeholder for handling non-geospatial data
    pass

def extract_non_geospatial_metadata(file_path):
    # Placeholder for handling non-geospatial data
    pass

def classify_non_geospatial_data(file_path):
    # Placeholder for handling non-geospatial data
    pass

def explore_and_find_duplicates(base_path):
    folder_times = defaultdict(float)
    unique_file_types = defaultdict(lambda: {'types': set(), 'size': 0})
    all_file_info = {}
    exact_duplicates = defaultdict(set)
    fuzzy_duplicates = defaultdict(set)
    metadata_info = set()
    lock = threading.Lock()

    def process_files(folder_name, file_paths):
        local_unique_file_types = {}
        local_metadata_info = set()
        for file_path in file_paths:
            if is_considered_file(file_path):
                start_time = time.time()
                file_type, file_size, mime_type, mime_encoding = get_file_info(file_path)

                if file_type == '.shp' or file_type == '.geojson' or file_type == '.kml':
                    gdf = ingest_geospatial_data(file_path)
                    metadata = extract_geospatial_metadata(gdf)
                    classification = classify_geospatial_data(gdf)
                    local_metadata_info.add((file_path, os.path.basename(file_path), file_type,
                                             convert_size(file_size), mime_type, mime_encoding,
                                             metadata, classification))
                else:
                    ingest_non_geospatial_data(file_path)
                    metadata = extract_non_geospatial_metadata(file_path)
                    classification = classify_non_geospatial_data(file_path)
                    local_metadata_info.add((file_path, os.path.basename(file_path), file_type,
                                             convert_size(file_size), mime_type, mime_encoding,
                                             metadata, classification))

                all_file_info[file_path] = {'types': set(), 'size': 0}
                all_file_info[file_path]['types'].add(file_type)
                all_file_info[file_path]['size'] += file_size

                local_unique_file_types[file_path] = {'types': set(), 'size': 0}
                local_unique_file_types[file_path]['types'].add(file_type)
                local_unique_file_types[file_path]['size'] += file_size

                end_time = time.time()
                elapsed_time = end_time - start_time
                folder_times[folder_name] += elapsed_time

        with lock:
            for file_path, file_info in local_unique_file_types.items():
                if file_path not in unique_file_types:
                    unique_file_types[file_path] = {'types': set(), 'size': 0}
                unique_file_types[file_path]['types'].update(file_info['types'])
                unique_file_types[file_path]['size'] += file_info['size']

        with lock:
            metadata_info.update(local_metadata_info)

    def process_folder(folder_path):
        for root, _, files in os.walk(folder_path):
            file_paths = [os.path.join(root, file) for file in files]
            process_files(os.path.basename(root), file_paths)

    def explore_compressed_folder(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if file.endswith(('.zip', '.tar', '.gz', '.bz2', '.rar', '.7z')):
                print(f"Exploring contents of: {file_path}")
                extract_folder = os.path.join(folder_path, os.path.splitext(file)[0])
                extract_archive(file_path, extract_folder)
                process_folder(extract_folder)
            else:
                process_folder(file_path)

    num_threads = min(multiprocessing.cpu_count() * 2, 32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(explore_compressed_folder, base_path),
            executor.submit(process_folder, base_path)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Error: {exc}")

        executor.shutdown(wait=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        hash_values = list(executor.map(hash_file, all_file_info.keys()))
        all_file_info_hashes = dict(zip(all_file_info.keys(), hash_values))

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

    metadata_df = pd.DataFrame(metadata_info, columns=['File', 'File Name', 'File Type', 'File Size', 'MIME Type', 'MIME Encoding', 'Metadata', 'Classification'])
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
    siamese_dataset_excel_filename = "siamese_dataset.xlsx"

    df_exact.to_excel(excel_exact_filename, index=False)
    df_fuzzy.to_excel(excel_fuzzy_filename, index=False)

    siamese_dataset = pd.DataFrame(columns=['File1Name', 'File2Name', 'HashFile1', 'HashFile2', 'Match'])

    for file_path1, duplicate_addresses in exact_duplicates.items():
        for file_path2 in duplicate_addresses:
            hash1 = all_file_info_hashes[file_path1]
            hash2 = all_file_info_hashes[file_path2]
            content_match = hash1 == hash2
            siamese_dataset = pd.concat([siamese_dataset, pd.DataFrame({
                'File1Name': [os.path.basename(file_path1)] * len(duplicate_addresses),
                'File2Name': [os.path.basename(file_path2)] * len(duplicate_addresses),
                'HashFile1': [hash1] * len(duplicate_addresses),
                'HashFile2': [hash2] * len(duplicate_addresses),
                'Match': [int(content_match)] * len(duplicate_addresses),
            })], ignore_index=True)

    for file_path1, duplicate_addresses in fuzzy_duplicates.items():
        for file_path2 in duplicate_addresses:
            hash1 = all_file_info_hashes[file_path1]
            hash2 = all_file_info_hashes[file_path2]
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

    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2

    folder_names = list(folder_times.keys())
    times = list(folder_times.values())

    plt.figure(figsize=(15, 8))
    plt.bar(folder_names, times, color='blue')
    plt.title(f'Time Taken for Each Folder and overall time taken is : {elapsed_time2}')
    plt.xlabel('Folder Name')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    print(f"\nTotal execution time: {elapsed_time1} seconds")

if __name__ == "__main__":
    base_path = r"D:\adhvik\adh\Hackathon\space hack\finals data\Topic-1\Geospatial Data Mangement"
    start_time1 = time.time()
    start_time2 = time.time()
    print("Timer started")
    explore_and_find_duplicates(base_path)
