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
path_vibration = os.path.join(base_dir, '../data/csv/vibration/Vibration_data_1.csv')
path_sound = os.path.join(base_dir, '../data/csv/soundsignal/Sound_data_1.csv')

df_vibration = pd.read_csv(path_vibration)
df_sound = pd.read_csv(path_sound)

# Reset index to avoid NaN in merged DataFrame
df_vibration.reset_index(drop=True, inplace=True)
df_sound.reset_index(drop=True, inplace=True)

# Merge DataFrames
df = pd.concat([df_vibration, df_sound], axis=1)

# Add Normalization
scaler = MinMaxScaler()
df[['Vibration (g)', 'Sound (dB)']] = scaler.fit_transform(df[['Vibration (g)', 'Sound (dB)']])

# Labeling Fault Class
def label_fault(row):
    if row['Vibration (g)'] < 0.5 and row['Sound (dB)'] < 0.5:
        return 'No Fault'
    elif (0.5 <= row['Vibration (g)'] <= 0.7) or (0.5 <= row['Sound (dB)'] <= 0.7):
        return 'Early Warning'
    else:
        return 'Critical Fault'

df['Fault_Class'] = df.apply(label_fault, axis=1)

# Features and Labels
X = df.drop(columns=['Fault_Class'])
y = df['Fault_Class'] 

# Feature Extraction: RMS, Amplitude, Frequency
def extract_features(signal):
    if len(signal) == 0:
        return 0, 0, 0  # Return zeros if the signal is empty
    
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
        return rms, amplitude, 0  # Return 0 for dominant frequency
    
    dominant_freq = xf[np.argmax(np.abs(yf[:N//2]))]  # Dominant Frequency
    
    return rms, amplitude, dominant_freq

# Handle NaN and Empty values
df.fillna(0, inplace=True)

# Apply Feature Extraction
features = []
for i in range(len(X)):
    vib_signal = X.loc[i, X.columns.str.contains('Vibration')].to_numpy()
    snd_signal = X.loc[i, X.columns.str.contains('Sound')].to_numpy()
    
    rms_vib, amp_vib, freq_vib = extract_features(vib_signal)
    rms_snd, amp_snd, freq_snd = extract_features(snd_signal)
    features.append([rms_vib, amp_vib, freq_vib, rms_snd, amp_snd, freq_snd])

# Create DataFrame for Features
features_df = pd.DataFrame(features, columns=[
    'RMS_Vibration', 'Amplitude_Vibration', 'Frequency_Vibration',
    'RMS_Sound', 'Amplitude_Sound', 'Frequency_Sound'
])

# Merge with Labels
X_features = features_df
y_labels = y

# Split Data
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
    output_path = os.path.join(output_dir, f'{model_name}_Compressor_confusion_matrix.png')
    plt.savefig(output_path)    
    plt.savefig(f'{model_name}_Compressor_confusion_matrix.png')
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
        
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_mean:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Final Evaluation
    print(f"\n{model_name} Final Test Accuracy:", acc)
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, model_name)


# Train and Evaluate - Fine Decision Tree with Cross-Validation
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
train_with_debug(dt_model, "Decision_Tree")

# Train and Evaluate - SVM Quadratic with Cross-Validation
svm_quad = SVC(kernel='poly', degree=2, random_state=42)
train_with_debug(svm_quad, "SVM_Quadratic")

# Train and Evaluate - SVM Cubic with Cross-Validation
svm_cubic = SVC(kernel='poly', degree=3, random_state=42)
train_with_debug(svm_cubic, "SVM_Cubic")


# Plot Frequency Spectrum for Sample Data
sample_index = 0 
num_samples = 100 

# DataFrame
vibation_signal = X.loc[sample_index : sample_index + num_samples, X.columns.str.contains('Vibration')].values.flatten()
vibation_signal = np.array(vibation_signal)

sound_signal = X.loc[sample_index : sample_index + num_samples, X.columns.str.contains('Sound')].values.flatten()
sound_signal = np.array(sound_signal)

# FFT และ Frequency Calculation
yf = fft(vibation_signal)
xf = fftfreq(len(vibation_signal), 1.0 / 100.0)[:len(vibation_signal)//2]

sound_yf = fft(sound_signal)
sound_xf = fftfreq(len(sound_signal), 1.0 / 100.0)[:len(sound_signal)//2]

# Plot Frequency Spectrum (1 Row, 2 Columns)
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Vibration Frequency Spectrum
axs[0].plot(xf, 2.0/len(vibation_signal) * np.abs(yf[:len(vibation_signal)//2]), color='blue')
axs[0].set_title('Frequency Spectrum (Vibration)')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Sound Frequency Spectrum
axs[1].plot(sound_xf, 2.0/len(sound_signal) * np.abs(sound_yf[:len(sound_signal)//2]), color='green')
axs[1].set_title('Frequency Spectrum (Sound)')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

plt.tight_layout()
plt.show()

