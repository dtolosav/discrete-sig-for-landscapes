import os
import numpy as np
import csv

# Directory containing the results
# results_dir = 'results'
results_dir = 'results/tests/'

# Output CSV file
# output_csv = 'results/kmeans_results_summary.csv'
output_csv = 'results/tests/kmeans_results_summary.csv'

# List to hold the data
data_rows = []

# Find all kmeans .npz files
for filename in os.listdir(results_dir):
    if filename.startswith('kmeans_') and filename.endswith('.npz'):
        filepath = os.path.join(results_dir, filename)
        try:
            # Load the .npz file
            data = np.load(filepath, allow_pickle=True)
            
            if 'summary_stats' in data:
                summary_stats = data['summary_stats'].item()  # Get the dict from the 0-d array
                # Extract the required values
                avg_ari = summary_stats['avg_ari']
                std_ari = summary_stats['std_ari']
                avg_nmi = summary_stats['avg_nmi']
                std_nmi = summary_stats['std_nmi']
            else:
                # Direct keys
                avg_ari = data['avg_ari']
                std_ari = data['std_ari']
                avg_nmi = data['avg_nmi']
                std_nmi = data['std_nmi']
            
            # Append to data rows
            data_rows.append({
                'filename': filename,
                'avg_ari': avg_ari,
                'std_ari': std_ari,
                'avg_nmi': avg_nmi,
                'std_nmi': std_nmi
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Write to CSV
if data_rows:
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'avg_ari', 'std_ari', 'avg_nmi', 'std_nmi']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data_rows:
            writer.writerow(row)
    
    print(f"Results written to {output_csv}")
else:
    print("No kmeans results found.")
