Data Redundancy Identification / Removal Algorithm

This project is done to solve real world problem of finding duplicate , unecessory files.

So this is how the Code works.

![sysarcpic ver final 2](https://github.com/AdhvikVithun/Space-Hack/assets/148479685/97b1f2b6-c533-49eb-83ba-78f0b9dec312)


This Python script is designed for Data Redundancy Removal / Identification. The script performs the following tasks:

Import Libraries:

os: Provides a way to interact with the operating system, such as reading file paths and checking file existence.
zipfile and tarfile: Used for extracting contents from zip and tar archives, respectively.
fuzzywuzzy: A library for fuzzy string matching, used here for comparing file names.
collections.defaultdict: A dictionary with default values, used to store file type information.
threading and concurrent.futures: Used for parallel processing of files and folders.
hashlib: Provides hashing algorithms for creating checksums.
pandas: A powerful data manipulation library.
mimetypes: Provides functions to guess the MIME type of a file.
time: Used to measure the execution time.
multiprocessing: Allows parallel processing using multiple processes.


Helper Functions:

fuzzy_match: Determines if two strings are similar based on a threshold.
get_file_info: Retrieves information about a file, such as extension, size, MIME type, and encoding.
is_considered_file: Checks if a given path points to a file.
convert_size: Converts file size to a human-readable format.
extract_archive: Extracts contents from supported archive types (zip and tar).
hash_file: Calculates the MD5 hash of a file.


Main Function (explore_and_find_duplicates):

Uses multithreading to explore the base folder and identify file types and sizes.
Extracts contents from compressed folders (e.g., zip, tar) and explores them.
Uses multiprocessing to parallelize the hashing process.
Compares files for exact and fuzzy duplicates based on file names and sizes.
Generates DataFrames for exact and fuzzy duplicates.
Prints metadata information about files.
Saves DataFrames and metadata information to Excel files.
Measures the execution time.


Streamlit Integration:

The script uses Streamlit to create a simple web app.
The title and an input field for the base path are displayed.
Upon clicking the "Find Duplicates" button, the explore_and_find_duplicates function is called, and the results are displayed using Streamlit's components.

Results Display:

Streamlit components (st.title, st.text_input, st.button, st.text, st.dataframe, st.pyplot) are used to display results and create interactive visualizations.
The exactly same and slightly similar files, along with metadata information, are presented in a tabular format.
A bar chart is created using Matplotlib and displayed to visualize the processing time for each folder with and without parallel processing.

Notes for GitHub:

Make sure to include relevant dependencies (e.g., fuzzywuzzy, pandas) in your requirements.txt or environment.yml file.
Provide clear documentation, a README file explaining how to use the script, and any prerequisites.
Consider adding comments to critical sections of the code to enhance readability.
Ensure that sensitive information, such as file paths, is properly handled (consider using configuration files or command-line arguments).
Include license information in your repository.

Usage:

Users can input the base path via the Streamlit web app.
Upon clicking the "Find Duplicates" button, the script processes the data and presents results interactively through the Streamlit interface.
Note:

This script provides a versatile tool for identifying redundant files in a specified directory and offers an intuitive web interface for users to interact with the results.
