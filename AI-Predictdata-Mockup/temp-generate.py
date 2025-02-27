import os
import numpy as np
import pandas as pd
csv_folder = None

def create_floder():
    global csv_folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = os.path.join(current_dir, "data", "csv", "temperature")
    
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

# Simulate temperature (°C) with a little noise (+) use a sine wave equation to simulate volatility
base_temperature = 34.5
temperature =  base_temperature + 4.0 * np.sin(2 * np.pi * time / (sampling_interval * 10)) + np.random.normal(0, 0.1, num_samples)


# Simulate humidity with a little noise (+) use a sine wave equation to simulate volatility
base_humidity = 40.0
humidity = base_humidity + (temperature - base_temperature) * 2.0 + np.random.normal(0, 0.5, num_samples)

df_temp = pd.DataFrame({
    "Time (ms)": time,
    "Temperature (°C)": temperature,
    "Humidity (%)": humidity
})

print(df_temp.head())
mockcsv_create(df_temp, "Temp_data")