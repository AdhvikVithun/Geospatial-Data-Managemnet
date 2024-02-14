import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from rtree import index
from shapely.geometry import Point

# Load latitude and longitude data from Excel file
df = pd.read_excel('D:/adhvik/adh/Hackathon/space hack/Data RR/hack code/testcluster.xlsx')

# Extract latitude and longitude columns
coordinates = df[['Latitude', 'Longitude']].values

# Standardize the data (optional but can be helpful for K-Means)
scaler = StandardScaler()
coordinates_scaled = scaler.fit_transform(coordinates)

# Define the number of clusters (you can adjust this based on your preference)
num_clusters = 34

# Explicitly set n_init and algorithm to potentially improve speed
kmeans = KMeans(n_clusters=num_clusters, n_init=10, algorithm='elkan', random_state=42)
clusters = kmeans.fit_predict(coordinates_scaled)

# Create Shapely geometry objects
df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

# Create R-Tree index
idx = index.Index()
for i, row in df.iterrows():
    idx.insert(i, row['geometry'].bounds)

# Build catalog system
catalog = df.groupby('State')['File'].apply(list).to_dict()

# Function to query GIS files based on coordinates
def query_files(query_latitude, query_longitude):
    query_point = Point(query_longitude, query_latitude)
    possible_matches = list(idx.intersection(query_point.bounds))
    state = None

    # Use R-Tree to find the state for the query coordinates
    for i in possible_matches:
        if df.loc[i, 'geometry'].contains(query_point):
            state = df.loc[i, 'State']
            break

    return catalog.get(state, [])

# Example query
query_latitude = 33.3731
query_longitude = 74.3089
result_files = query_files(query_latitude, query_longitude)
print("\nFiles in the given location:", result_files)
