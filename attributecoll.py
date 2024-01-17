import os
import geopandas as gpd
import pandas as pd

# Replace this with the path to your directory containing shapefiles
directory_path = r'D:\adhvik\adh\Hackathon\space hack\finals data\Topic-1\Geospatial Data Mangement\Geospatial Data Mangement'

# Create an empty list to store data
data_list = []

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.shp'):
        # Form the complete path to the shapefile
        file_path = os.path.join(directory_path, filename)

        # Load the shapefile using GeoPandas
        gdf = gpd.read_file(file_path)

        # Iterate through each feature in the GeoDataFrame
        for index, row in gdf.iterrows():
            # Access the geometry of the feature
            geometry = row['geometry']

            # Extract latitude and longitude if the geometry is a Point
            latitude = geometry.y if geometry.geom_type == 'Point' else None
            longitude = geometry.x if geometry.geom_type == 'Point' else None
            
            centroid = geometry.centroid if geometry.is_valid else None

            # Only append data if both latitude and longitude are present
            if latitude or longitude is not None:
                # Append data to the list
                data_list.append({
                    'File': filename,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'Centroid': centroid
                })

# Create a DataFrame from the list
df = pd.DataFrame(data_list)

# Save the DataFrame to an Excel file
excel_file_path = 'output_data_filtered.xlsx'  # Change this to the desired output file path
df.to_excel(excel_file_path, index=False)
print(f"Filtered data saved to {excel_file_path}")
