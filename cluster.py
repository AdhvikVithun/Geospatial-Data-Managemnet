import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim

# Load latitude and longitude data from Excel file
df = pd.read_excel('output_data_merged.xlsx')

# Extract latitude and longitude columns
coordinates = df[['Latitude', 'Longitude']].values[:10000]  # Use the first 10,000 rows

# Standardize the data (optional but can be helpful for K-Means)
scaler = StandardScaler()
coordinates_scaled = scaler.fit_transform(coordinates)

# Define the number of clusters (you can adjust this based on your preference)
num_clusters = 28

# Explicitly set n_init and algorithm to potentially improve speed
kmeans = KMeans(n_clusters=num_clusters, n_init=10, algorithm='elkan', random_state=42)
clusters = kmeans.fit_predict(coordinates_scaled)

# Create a folium map centered at the first coordinate
center_lat, center_long = coordinates[0]
map_clusters = folium.Map(location=[center_lat, center_long], zoom_start=6)

# Create a MarkerCluster for each cluster
marker_cluster = MarkerCluster().add_to(map_clusters)

# Add markers to the MarkerCluster for each data point with cluster information
for i in range(len(clusters)):
    lat, long = coordinates[i]
    cluster_label = f'Lat: {lat}, Long: {long}'
    
    folium.Marker([lat, long], popup=cluster_label, 
                  icon=folium.Icon(color=f'lightblue')).add_to(marker_cluster)

map_clusters.save('cluster_map.html')
print("Clustering completed. Map saved to cluster_map.html")

# Reverse geocode to get state information for each cluster
geolocator = Nominatim(user_agent="geo_locator")
state_info = []

for i in range(num_clusters):
    cluster_indices = np.where(clusters == i)[0]
    cluster_data = df.iloc[cluster_indices]
    
    center_lat, center_long = cluster_data[['Latitude', 'Longitude']].mean()
    location = geolocator.reverse((center_lat, center_long), language='en')
    
    # Check if geocoding was successful before accessing attributes
    if location and hasattr(location, 'raw') and 'address' in location.raw and 'state' in location.raw['address']:
        state = location.raw['address']['state']
    else:
        state = None
    
    state_info.append({'Cluster': i, 'State': state})

state_df = pd.DataFrame(state_info)
print(state_df)
