import os
import numpy as np
import pandas as pd
csv_folder = None

def create_floder():
    global csv_folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = os.path.join(current_dir, "data", "csv", "soundsignal")
    
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


# Simulate Sound-Signal Sensor with a little noise (+) use a sine wave equation to simulate volatility
sound = 0.5 + 0.2 * np.sin(2 * np.pi * time / (sampling_interval * 15)) + \
        np.random.normal(0, 0.05, num_samples)

df_sound_signal = pd.DataFrame({
    "Time (s)": time,
    "Sound (dB)": sound
})

print(df_sound_signal.head())
mockcsv_create(df_sound_signal, "Sound_data")
