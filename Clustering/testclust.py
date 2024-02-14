import pandas as pd
from sklearn.cluster import KMeans
from rtree import index
from geopy.distance import geodesic
import joblib

# Read the input data from the Excel file
file_path = r'D:\adhvik\adh\Hackathon\space hack\Data RR\hack code\output_data_merged.xlsx'
df = pd.read_excel(file_path)

print("Checking if the Model is already trained")
# Check if the model is already trained and saved
model_file_path = r'D:\adhvik\adh\Hackathon\space hack\Data RR\hack code\kmeans_model.joblib'
try:
    kmeans = joblib.load(model_file_path)
    
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        print("Retraining the model as 'Cluster' column is missing.")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(df[['Longitude', 'Latitude']])
        joblib.dump(kmeans, model_file_path)
except FileNotFoundError:
    # If the model file is not found, train the model and save it
    print("Model file is not found so training the model")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Longitude', 'Latitude']])
    joblib.dump(kmeans, model_file_path)

# Create a spatial index
spatial_index = index.Index()
for i, row in df.iterrows():
    spatial_index.insert(i, (row['Longitude'], row['Latitude'], row['Longitude'], row['Latitude']))

# Function to find the nearest cluster based on geodesic distance using spatial indexing
def find_nearest_files(latitude, longitude, df, spatial_index):
    point = (latitude, longitude)
    nearest_index = next(spatial_index.nearest((point[0], point[1], point[0], point[1]), 1))
    nearest_cluster = df.at[nearest_index, 'Cluster']  # Replace 'State_Cluster' with 'Cluster'
    
    # Get files in the nearest cluster
    nearest_files = df[df['Cluster'] == nearest_cluster]['File'].unique()
    
    return nearest_files

# Example: Find the files for a given location
query_latitude = 33.3731  
query_longitude = 74.3089
nearest_files = find_nearest_files(query_latitude, query_longitude, df, spatial_index)

print("Nearest Files:", nearest_files)