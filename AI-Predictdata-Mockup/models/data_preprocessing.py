# data_preprocessing.py

# Import Libraries
import os
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

# Load Data Function
def load_data():
    base_dir = os.path.dirname(__file__)
    path_temp = os.path.join(base_dir, '../data/csv/temperature/Temp_data_1.csv')
    path_airpressure = os.path.join(base_dir, '../data/csv/airpressure/Airpress_data_1.csv')
    path_refri_flow = os.path.join(base_dir, '../data/csv/refrigerant/Ref_Flowdata_1.csv')
    path_refri_pressure = os.path.join(base_dir, '../data/csv/refrigerant/Ref_Pressuredata_1.csv')

    df_temp = pd.read_csv(path_temp)
    df_airpressure = pd.read_csv(path_airpressure)
    df_refri_flow = pd.read_csv(path_refri_flow)
    df_refri_pressure = pd.read_csv(path_refri_pressure)

    # Reset index to avoid NaN in merged DataFrame
    df_temp.reset_index(drop=True, inplace=True)
    df_airpressure.reset_index(drop=True, inplace=True)
    df_refri_flow.reset_index(drop=True, inplace=True)
    df_refri_pressure.reset_index(drop=True, inplace=True)

    # Merge DataFrames
    df = pd.concat([df_temp, df_airpressure, df_refri_flow, df_refri_pressure], axis=1)
    df.fillna(0, inplace=True)  # Handle NaN and Empty values
    
    return df

# Feature Extraction Function
def extract_features(signal):
    if len(signal) == 0:
        return 0, 0, 0, 0  # Return zeros if the signal is empty

    # Mean
    mean_val = np.mean(signal)
    
    # RMS Calculation
    rms = np.sqrt(np.mean(np.square(signal)))
    
    # Amplitude (Peak-to-Peak)
    amplitude = np.max(signal) - np.min(signal)
    
    # Frequency using FFT
    N = len(signal)
    T = 1.0 / 100.0  # Sampling interval (Example: 100 Hz)
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    
    # Check if the frequency array is empty
    if len(xf) == 0:
        return mean_val, rms, amplitude, 0  # Return 0 for dominant frequency
    
    dominant_freq = xf[np.argmax(np.abs(yf[:N//2]))]  # Dominant Frequency
    
    return mean_val, rms, amplitude, dominant_freq

# Improved Labeling Function
def label_fault(row):
    # Thresholds (ปรับตามลักษณะข้อมูลที่ได้)
    temp_threshold = [30, 38]  # [Normal, Warning, Critical]
    flow_threshold = [40, 60]
    press_threshold = [600, 750]

    # Check Conditions
    if (
        (temp_threshold[0] <= row['Mean_Temp'] <= temp_threshold[1]) and
        (flow_threshold[0] <= row['Mean_Flow'] <= flow_threshold[1]) and
        (press_threshold[0] <= row['Mean_Press'] <= press_threshold[1])
    ):
        return 'No Fault'
    
    elif (
        (temp_threshold[1] < row['Mean_Temp'] <= temp_threshold[1] + 2) or
        (flow_threshold[1] < row['Mean_Flow'] <= flow_threshold[1] + 5) or
        (press_threshold[1] < row['Mean_Press'] <= press_threshold[1] + 50)
    ):
        return 'Early Warning'
    
    else:
        return 'Critical Fault'

# Extract Features for Group 1
def get_features():
    df = load_data()
    features = []
    for i in range(len(df)):
        temp_signal = df.loc[i, df.columns.str.contains('Temp')].to_numpy()
        air_signal = df.loc[i, df.columns.str.contains('Air')].to_numpy()
        ref_flow_signal = df.loc[i, df.columns.str.contains('Flow')].to_numpy()
        ref_press_signal = df.loc[i, df.columns.str.contains('Press')].to_numpy()
        
        temp_features = extract_features(temp_signal)
        air_features = extract_features(air_signal)
        ref_flow_features = extract_features(ref_flow_signal)
        ref_press_features = extract_features(ref_press_signal)
        
        combined_features = temp_features + air_features + ref_flow_features + ref_press_features
        features.append(combined_features)

    feature_columns = [
        'Mean_Temp', 'RMS_Temp', 'Amp_Temp', 'Freq_Temp',
        'Mean_Air', 'RMS_Air', 'Amp_Air', 'Freq_Air',
        'Mean_Flow', 'RMS_Flow', 'Amp_Flow', 'Freq_Flow',
        'Mean_Press', 'RMS_Press', 'Amp_Press', 'Freq_Press'
    ]
    features_df = pd.DataFrame(features, columns=feature_columns)

    # Labeling Data
    features_df['Fault_Label'] = features_df.apply(label_fault, axis=1)

    # Save to CSV for Analysis
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, '../data/output/labeled_features.csv')
    features_df.to_csv(output_path, index=False)
    
    print(f"CSV Saved: {output_path}")

    return features_df

# Main Execution
if __name__ == "__main__":
    get_features()