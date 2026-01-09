import kagglehub
import os
import shutil

print("Starting download script...")

try:
    print("Attempting kagglehub download...")
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
    print(f"Download completed. Path: {path}")
    
    src_file = os.path.join(path, 'Clean_Dataset.csv')
    dst_file = 'Clean_Dataset.csv'
    
    if os.path.exists(src_file):
        print(f"Found file at {src_file}. Copying to current directory...")
        shutil.copy(src_file, dst_file)
        print("Copy successful.")
    else:
        print(f"File Clean_Dataset.csv not found in {path}")
        
except Exception as e:
    print(f"Error during download: {e}")

print("Script finished.")
