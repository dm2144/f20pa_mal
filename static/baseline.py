import os
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def load_and_flatten(directory, label):
    #Loads .npy embeddings and averages them into a single 64-dim vector.
    files = glob.glob(os.path.join(directory, "*.npy"))
    features = []
    labels = []
    
    for f in files:
        try:
            data = np.load(f) # Shape: [Nodes, 64]
            if data.ndim == 2:
                mean_feature = np.mean(data, axis=0) #we take the mean across all nodes to represent the file
                features.append(mean_feature)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return np.array(features), np.array(labels)

def run_baselines(base_path):
    # Setup paths based on your structure
    n_b_path = os.path.join(base_path, "nodes/n_b")
    n_m_path = os.path.join(base_path, "nodes/n_m")
    
    print(f" Loading data from {base_path} ")
    X_b, y_b = load_and_flatten(n_b_path, 0) # 0 for Benign
    X_m, y_m = load_and_flatten(n_m_path, 1) # 1 for Malware
    
    X = np.concatenate([X_b, X_m])
    y = np.concatenate([y_b, y_m])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Random Forest
    print("\n[MODEL] Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(classification_report(y_test, rf_preds, target_names=['Benign', 'Malware']))

    # 2. SVM
    print("\n[MODEL] Support Vector Machine (SVM)")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, svm_preds):.4f}")
    print(classification_report(y_test, svm_preds, target_names=['Benign', 'Malware']))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 baseline.py <path_to_analysis_root>") #for ref
    else:
        run_baselines(sys.argv[1])
