import os
import numpy as np
import pandas as pd
csv_folder = None

def create_floder():
    global csv_folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = os.path.join(current_dir, "data", "csv", "refrigerant")
    
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
time = np.arange(0, num_samples * sampling_interval, sampling_interval) # Sampling every 5 ms


# Simulate refrigerant pressure using base value with random oscillation
refrigerant_pressure = 200 + np.random.uniform(-10, 10, num_samples)

# Simulate the refrigerant flow rate using a sine wave mixed with noise
refrigerant_flow = 50 + 5 * np.sin(2 * np.pi * time / (sampling_interval * 25)) + np.random.normal(0, 3, num_samples)


df_refri_pressure = pd.DataFrame({
    "Time (s)": time,
    "Refrigerant Pressure (psi)": refrigerant_pressure
})

df_refri_flow = pd.DataFrame({
    "Time (s)": time,
    "Refrigerant Flow (L/min)": refrigerant_flow
})

print(df_refri_pressure.head())
mockcsv_create(df_refri_pressure, "Ref_Pressuredata")

print(df_refri_flow.head())
mockcsv_create(df_refri_flow, "Ref_Flowdata")