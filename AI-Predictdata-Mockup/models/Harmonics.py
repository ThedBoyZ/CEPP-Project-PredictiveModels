# Import Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns


base_dir = os.path.dirname(__file__)
path_current = os.path.join(base_dir, '../data/csv/current/Current_data_1.csv')

df_current = pd.read_csv(path_current)

#  Handle NaN and Empty values
df_current.fillna(0, inplace=True) 

scaler = MinMaxScaler()
df_current['Current (A)'] = scaler.fit_transform(df_current[['Current (A)']])

def label_fault_current(row):
    if row['Current (A)'] < 0.5:
        return 'No Fault'
    elif 0.5 <= row['Current (A)'] <= 0.7:
        return 'Early Warning'
    else:
        return 'Critical Fault'

df_current['Fault_Class'] = df_current.apply(label_fault_current, axis=1)

X = df_current.drop(columns=['Fault_Class']) 
y = df_current['Fault_Class']            

def extract_features_current(signal):
    if len(signal) == 0:
        print("Warning: Empty Signal Passed to extract_features_current().")
        return 0, 0, 0, 0
    
    signal = np.nan_to_num(signal)
    
    # RMS Current, Mean Current Current Fluctuation (Amplitude)
    rms = np.sqrt(np.mean(signal**2))      
    mean_current = np.mean(signal)      
    amplitude = np.max(signal) - np.min(signal)      
    
    # Frequency Component using FFT
    yf = fft(signal)
    xf = fftfreq(len(signal), 1.0 / 100.0)[:len(signal)//2]
    
    harmonic_spectrum = 2.0/len(signal) * np.abs(yf[:len(signal)//2])
    if len(harmonic_spectrum) > 0:
        dominant_freq = xf[np.argmax(harmonic_spectrum)]  # Harmonics / Dominant Frequency
    else:
        dominant_freq = 0
    
    return rms, mean_current, amplitude, dominant_freq

features = []
for i in range(len(X)):
    current_signal = X.loc[i, X.columns.str.contains('Current')].to_numpy()
    current_signal = current_signal[~np.isnan(current_signal)]
    current_signal = np.array(current_signal).flatten()
    
    # Check ถif current_signal to Empty skip
    if len(current_signal) == 0:
        print(f"Warning: Empty Current Signal at Index {i}. Skipping...")
        continue
    
    # Extract Features for Current
    rms_cur, mean_cur, amp_cur, freq_cur = extract_features_current(current_signal)
    features.append([rms_cur, mean_cur, amp_cur, freq_cur])

# Create DataFrame for Features
features_df = pd.DataFrame(features, columns=[
    'RMS_Current', 'Mean_Current', 'Amplitude_Current', 'Frequency_Current'
])

# Merge with Labels
X_features = features_df
y_labels = y

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.3, random_state=42)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    if model_name == "Decision_Tree":
        output_dir = os.path.join(base_dir, '../data/output/DT')
    elif model_name == "SVM_Cubic":
        output_dir = os.path.join(base_dir, '../data/output/SVM_Cubic')
    elif model_name == "SVM_Quadratic":
        output_dir = os.path.join(base_dir, '../data/output/SVM_Quadratic')
    else:
        output_dir = os.path.join(base_dir, '../data/output/Unknown_Model')
        print(f"Warning: Unknown model_name '{model_name}', saving to {output_dir}")
    output_path = os.path.join(output_dir, f'{model_name}_Current_confusion_matrix.png')
    plt.savefig(output_path)    
    plt.close()

def train_with_debug(model, model_name, num_epochs=32, k_splits=5):
    print(f"\n🔧 Training {model_name} Model with Cross-Validation...\n")
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    
    for epoch in range(num_epochs):
        # Cross-Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)

        print(f"\nEpoch {epoch+1}/{num_epochs} - Accuracy: {acc:.4f}")
        
    print(f"Cross-Validation Scores: {cv_scores.mean()}")
    print(f"Mean Cross-Validation Accuracy: {cv_mean:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    print(f"\n{model_name} Final Test Accuracy:", acc)
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, model_name)


# Train and Evaluate - Fine Decision Tree for Current
dt_model_current = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
train_with_debug(dt_model_current, "Decision_Tree")

# Train and Evaluate - SVM Quadratic for Current
svm_quad_current = SVC(kernel='poly', degree=2, random_state=42)
train_with_debug(svm_quad_current, "SVM_Quadratic")

# Train and Evaluate - SVM Cubic for Current
svm_cubic_current = SVC(kernel='poly', degree=3, random_state=42)
train_with_debug(svm_cubic_current, "SVM_Cubic")


# Plot Frequency Spectrum for Sample Data
sample_index = 0  
num_samples = 100 

# DataFrame 
current_signal = X.loc[sample_index : sample_index + num_samples, X.columns.str.contains('Current')].values.flatten()
current_signal = np.array(current_signal)

# FFT และ Frequency Calculation สำหรับ Current
yf = fft(current_signal)
xf = fftfreq(len(current_signal), 1.0 / 100.0)[:len(current_signal)//2]

plt.figure(figsize=(12, 6))
plt.plot(xf, 2.0/len(current_signal) * np.abs(yf[:len(current_signal)//2]), color='red')
plt.title('Frequency Spectrum (Current) - Harmonics')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

# 🔥 Mark Harmonics - เลือก Harmonics ที่ต้องการแสดง
harmonics = [25, 50]  # ตัวอย่าง Harmonics ของมอเตอร์ไฟฟ้า
for h in harmonics:
    plt.axvline(x=h, color='green', linestyle='--', alpha=0.7)
    plt.text(h, plt.ylim()[1]*0.8, f'{h} Hz', color='green', fontsize=10)

plt.show()