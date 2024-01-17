import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster

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

# Create a folium map centered at the first coordinate
center_lat, center_long = state_coordinates[list(state_coordinates.keys())[0]][0]
map_clusters = folium.Map(location=[center_lat, center_long], zoom_start=6)

# Create a MarkerCluster for each state
for state, coordinates in state_coordinates.items():
    marker_cluster = MarkerCluster().add_to(map_clusters)

    X = np.array(coordinates)

    # Standardize the data (optional but can be helpful for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the number of clusters (you can adjust this based on your preference)
    num_clusters = 3

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add markers to the MarkerCluster for each feature in the state
    for i, feature in enumerate(geojson_data['features']):
        if feature['properties']['state'] == state and i < len(clusters):  # index within bounds
            cluster = int(clusters[i])
            lat, long = feature['geometry']['coordinates'][0], feature['geometry']['coordinates'][1]
            folium.Marker([lat, long], popup=f'Cluster: {cluster}', icon=folium.Icon(color=f'lightblue')).add_to(marker_cluster)


map_clusters.save('cluster_map.html')
print("Clustering completed. Map saved to cluster_map.html")
