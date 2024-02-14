import os
import geopandas as gpd
import pandas as pd
import json
from osgeo import gdal
import csv

# Replace this with the path to your directory containing files of different formats
directory_path = r'D:\adhvik\adh\Hackathon\space hack\finals data\Topic-1\Geospatial Data Mangement\Geospatial Data Mangement'

# Create an empty list to store data
data_list = []

# Create a dictionary to store coordinates for each state
state_coordinates = {}

# Supported file extensions
supported_extensions = ['.gpkg', '.cpg', '.flt', '.csv', '.tif', '.txt', '.prj', '.shx', '.xml', '.shp', '.gml', '.kml', '.json', '.geojson']

# Helper function to get latitudes and longitudes from GeoTIFF files
def get_lat_lon_from_tif(tif_file_path):
    dataset = gdal.Open(tif_file_path)
    if not dataset:
        print(f"Error: Unable to open the GeoTIFF file: {tif_file_path}")
        return [], []
    
    geotransform = dataset.GetGeoTransform()
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    latitudes = [geotransform[3] + i * geotransform[5] for i in range(height)]
    longitudes = [geotransform[0] + j * geotransform[1] for j in range(width)]

    return latitudes, longitudes

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    # Check if the file has a supported extension
    _, extension = os.path.splitext(filename)
    if extension.lower() in supported_extensions:
        try:
            # Load data from the file based on its extension
            if extension.lower() == '.shp':
                gdf = gpd.read_file(file_path)
                for index, row in gdf.iterrows():
                    geometry = row['geometry']
                    latitude = geometry.y if geometry.geom_type == 'Point' else None
                    longitude = geometry.x if geometry.geom_type == 'Point' else None

                    if latitude is not None and longitude is not None:
                        data_list.append({
                            'File': filename,
                            'Latitude': latitude,
                            'Longitude': longitude
                        })
            elif extension.lower() in ['.gpkg', '.gml', '.kml', '.geojson']:
                # Handle other formats using GeoPandas
                gdf = gpd.read_file(file_path)
                for index, row in gdf.iterrows():
                    geometry = row['geometry']
                    latitude = geometry.y if geometry.geom_type == 'Point' else None
                    longitude = geometry.x if geometry.geom_type == 'Point' else None

                    if latitude is not None and longitude is not None:
                        data_list.append({
                            'File': filename,
                            'Latitude': latitude,
                            'Longitude': longitude
                        })
            elif extension.lower() == '.csv':
                # Handle CSV using pandas
                df = pd.read_csv(file_path)
                for index, row in df.iterrows():
                    latitude = row['latitude']
                    longitude = row['longitude']

                    if latitude is not None and longitude is not None:
                        data_list.append({
                            'File': filename,
                            'Latitude': latitude,
                            'Longitude': longitude
                        })
            elif extension.lower() == '.tif':
                    # Handle GeoTIFF files
                    dataset = gdal.Open(file_path)
    
                    if not dataset:
                        print(f"Error: Unable to open the GeoTIFF file: {file_path}")
                        continue

                    # Get geotransform information
                    geotransform = dataset.GetGeoTransform()

                    # Get raster size
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize

                    # Extract latitudes and longitudes
                    latitudes = [geotransform[3] + i * geotransform[5] for i in range(height)]
                    longitudes = [geotransform[0] + j * geotransform[1] for j in range(width)]

                    #                Write data to the DataFrame
                    for lat, lon in zip(latitudes, longitudes):
                        data_list.append({
                        'File': filename,
                        'Latitude': lat,
                        'Longitude': lon
                        })
    
            elif extension.lower() in ['.json', '.geojson']:
                # Handle GeoJSON format
                with open(file_path, 'r') as f:
                    geojson_data = json.load(f)

                # Extract coordinates from GeoJSON features
                for feature in geojson_data['features']:
                    state = feature['properties']['state']
                    long = feature['geometry']['coordinates'][0]
                    lat = feature['geometry']['coordinates'][1]

                    # Add coordinates to the state in the dictionary
                    if state not in state_coordinates:
                        state_coordinates[state] = []
                    state_coordinates[state].append({'Latitude': lat, 'Longitude': long})

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

# Create a DataFrame from the list
df = pd.DataFrame(data_list)

# Save the DataFrame to an Excel file
excel_file_path = 'output_data_filtered.xlsx'  # Change this to the desired output file path
df.to_excel(excel_file_path, index=False)
print(f"Filtered data saved to {excel_file_path}")

# Optionally, print state coordinates
for state, coordinates in state_coordinates.items():
    print(f"State: {state}")
    print(f"Coordinates: {coordinates}")
