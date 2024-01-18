import pandas as pd

# Read the CSV file into a pandas DataFrame
csv_file_path = r'D:\adhvik\adh\Hackathon\space hack\Data RR\hack code\final.csv'
csv_data = pd.read_csv(csv_file_path)

# Read the Excel file into a pandas DataFrame
excel_file_path = r'D:\adhvik\adh\Hackathon\space hack\Data RR\hack code\output_data_filtered.xlsx'
excel_data = pd.read_excel(excel_file_path)

# Concatenate the two DataFrames vertically
merged_data = pd.concat([excel_data, csv_data[['File', 'Longitude', 'Latitude']]])

# Save the merged DataFrame to a new Excel file
output_file_path = r'D:\adhvik\adh\Hackathon\space hack\Data RR\hack code\output_data_merged.xlsx'
merged_data.to_excel(output_file_path, index=False)

print("Merged data saved to:", output_file_path)

