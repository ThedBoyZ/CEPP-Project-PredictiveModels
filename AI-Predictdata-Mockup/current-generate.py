import os
import numpy as np
import pandas as pd
csv_folder = None

def create_floder():
    global csv_folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = os.path.join(current_dir,  "data", "csv", "current")
    
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
sampling_interval = 5000  # 5 ms sampling
time = np.arange(0, num_samples * sampling_interval, sampling_interval)  # Sampling every 5 ms

# Simulate Current with Harmonics and Noise
fundamental_freq = 50  # Main frequency at 50 Hz
harmonics = [2, 3, 5]  # Harmonic Multiples
amplitudes = [0.2, 0.15, 0.1]  # Amplitudes for Harmonics

# Base Current Signal with Noise
current = 1.0 + 0.2 * np.sin(2 * np.pi * fundamental_freq * time / 1e6)
current += np.random.normal(0, 0.1, num_samples)  # White Noise

# Adding Harmonics for more fluctuations
for i, harmonic in enumerate(harmonics):
    current += amplitudes[i] * np.sin(2 * np.pi * fundamental_freq * harmonic * time / 1e6)

# Optional: Add Random Spikes to simulate transient noise
spike_indices = np.random.choice(num_samples, size=10, replace=False)
current[spike_indices] += np.random.uniform(-0.5, 0.5, size=10)

# Create DataFrame
df_current = pd.DataFrame({
    "Time (s)": time,
    "Current (A)": current
})

print(df_current.head())
mockcsv_create(df_current, "Current_data")