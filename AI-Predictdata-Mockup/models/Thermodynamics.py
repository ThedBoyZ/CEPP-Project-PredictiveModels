# Import Libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.dirname(__file__)

 # Import the get_features function
from data_preprocessing import get_features 
features_df = get_features() 

# Split Features and Labels
X = features_df.drop(columns=['Fault_Label'])
y = features_df['Fault_Label']

# Normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
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
    output_path = os.path.join(output_dir, f'{model_name}_Thermo_confusion_matrix.png')
    plt.savefig(output_path)    
    plt.close()

def train_with_debug(model, model_name, num_epochs=32, k_splits=5):
    print(f"\n🔧 Training {model_name} Model with Cross-Validation...\n")
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    
    for epoch in range(num_epochs):
        # Cross-Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        
        # Train Model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate Accuracy
        acc = accuracy_score(y_test, y_pred)

        print(f"\nEpoch {epoch+1}/{num_epochs} - Accuracy: {acc:.4f}")
        
    # Debug Message
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

# Train and Evaluate - SVM (Quadratic)
svm_quad = SVC(kernel='poly', degree=2, C=1, random_state=42)
train_with_debug(svm_quad, "SVM_Quadratic")

# Train and Evaluate - SVM (Cubic)
svm_cubic = SVC(kernel='poly', degree=3, C=1, random_state=42)
train_with_debug(svm_cubic, "SVM_Cubic")


# 🔥 Plot 2 Rows, 2 Columns 
base_dir = os.path.dirname(__file__)
path_temp = os.path.join(base_dir, '../data/csv/temperature/Temp_data_1.csv')
path_airpressure = os.path.join(base_dir, '../data/csv/airpressure/Airpress_data_1.csv')
path_refri_flow = os.path.join(base_dir, '../data/csv/refrigerant/Ref_Flowdata_1.csv')
path_refri_pressure = os.path.join(base_dir, '../data/csv/refrigerant/Ref_Pressuredata_1.csv')

df_temp = pd.read_csv(path_temp)
df_air = pd.read_csv(path_airpressure)
df_ref_flow = pd.read_csv(path_refri_flow)
df_ref_press = pd.read_csv(path_refri_pressure)

print("Temp Columns:", df_temp.columns)
print("Air Columns:", df_air.columns)
print("Ref Flow Columns:", df_ref_flow.columns)
print("Ref Pressure Columns:", df_ref_press.columns)

# Numpy Array Change Column desired
temp_signal = df_temp.iloc[:, 1].values  
air_signal = df_air.iloc[:, 1].values
ref_flow_signal = df_ref_flow.iloc[:, 1].values
ref_press_signal = df_ref_press.iloc[:, 1].values

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Signal 1 (Temperature)
axs[0, 0].plot(temp_signal, color='red')
axs[0, 0].set_title('Temperature Signal')
axs[0, 0].set_xlabel('Sample')
axs[0, 0].set_ylabel('Value')
axs[0, 0].grid(True)

# Signal 2 (Air Pressure)
axs[0, 1].plot(air_signal, color='blue')
axs[0, 1].set_title('Air Pressure Signal')
axs[0, 1].set_xlabel('Sample')
axs[0, 1].set_ylabel('Value')
axs[0, 1].grid(True)

# Signal 3 (Refrigerant Flow)
axs[1, 0].plot(ref_flow_signal, color='green')
axs[1, 0].set_title('Refrigerant Flow Signal')
axs[1, 0].set_xlabel('Sample')
axs[1, 0].set_ylabel('Value')
axs[1, 0].grid(True)

# Signal 4 (Refrigerant Pressure)
axs[1, 1].plot(ref_press_signal, color='purple')
axs[1, 1].set_title('Refrigerant Pressure Signal')
axs[1, 1].set_xlabel('Sample')
axs[1, 1].set_ylabel('Value')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()