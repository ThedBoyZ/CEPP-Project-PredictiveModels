import os
import numpy as np
import pandas as pd
csv_folder = None

def create_floder():
    global csv_folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = os.path.join(current_dir, "data", "csv", "airpressure")
    
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        print("Created folder:", csv_folder)
    else:
        print("Folder already exists:", csv_folder)
    
def mockcsv_create(df, filename):
    global csv_folder
    if csv_folder is None:
        print("CSV folder is not set. Please call create_folder() first.")
        return
        
    base_filename = filename
    extension = ".csv"
    version = 1
    filename = os.path.join(csv_folder, f"{base_filename}_{version}{extension}")

    while os.path.exists(filename):
        version += 1
        filename = os.path.join(csv_folder, f"{base_filename}_{version}{extension}")

    df.to_csv(filename, index=False)
    print(f"Saved file: {filename}")

create_floder()    
# Number Sampling
num_samples = 1000
sampling_interval = 5000 # 5s sampling 
time = np.arange(0, num_samples * sampling_interval, sampling_interval) # Sampling every 10 ms

# Simulate Air Pressure (hPa) with little noise
pressure = 1013.25 - (time / (sampling_interval * 100)) + np.random.normal(0, 1.5, num_samples)

df_airp = pd.DataFrame({
    "Time (ms)": time,
    "Pressure (hPa)": pressure
})

print(df_airp.head())
mockcsv_create(df_airp, "Airpress_data")