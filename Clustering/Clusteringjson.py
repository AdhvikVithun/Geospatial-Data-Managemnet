import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load GeoJSON data from file
with open('geojson1.geojson', 'r') as file:
    geojson_data = json.load(file)

# Create a dictionary to store coordinates for each state
state_coordinates = {}

# Extract coordinates from GeoJSON features
for feature in geojson_data['features']:
    state = feature['properties']['state']
    lat = feature['geometry']['coordinates'][0]
    long = feature['geometry']['coordinates'][1]

    # Add coordinates to the state in the dictionary
    if state not in state_coordinates:
        state_coordinates[state] = []
    state_coordinates[state].append([lat, long])

# Perform clustering for each state
for state, coordinates in state_coordinates.items():
    X = np.array(coordinates)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the number of clusters (you can adjust this based on your preference)
    num_clusters = 3

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add the cluster information to the GeoJSON properties for each feature in the state
    for i, feature in enumerate(geojson_data['features']):
        if feature['properties']['state'] == state:
            feature['properties']['cluster'] = int(clusters[i])

# Save the updated GeoJSON data to a new file
with open('geojson_clusters.geojson', 'w') as output_file:
    json.dump(geojson_data, output_file, indent=2)

print("Clustering completed. Results saved to geojson_clusters.geojson")
