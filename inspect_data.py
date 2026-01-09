import pandas as pd
import os
import kagglehub

try:
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
    file_path = os.path.join(path, 'Clean_Dataset.csv')
    df = pd.read_csv(file_path)
    print("Columns:", df.columns.tolist())
    print("Classes:", df['class'].unique())
    print("Stops:", df['stops'].unique())
except Exception as e:
    print(e)
