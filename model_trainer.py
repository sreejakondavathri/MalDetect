import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def generate_synthetic_dataset(num_samples=200, output_path='data/features.csv'):
    """
    Generate synthetic training data
    Returns DataFrame
    """
    print(f"Generating {num_samples} synthetic samples...")
    
    data = []
    
    # Generate BENIGN file characteristics
    for i in range(num_samples // 2):
        features = {
            'file_name': f'benign_{i}.exe',
            'file_size': np.random.randint(10000, 500000),
            'entropy': np.random.uniform(4.0, 6.5),
            'num_sections': np.random.randint(2, 5),
            'has_pe_signature': 1,
            'num_strings': np.random.randint(50, 300),
            'avg_string_length': np.random.uniform(8, 20),
            'has_VirtualAlloc': 0,
            'has_VirtualProtect': 0,
            'has_CreateRemoteThread': 0,
            'has_WriteProcessMemory': 0,
            'total_suspicious_apis': 0,
            'label': 0  # Benign
        }
        data.append(features)
    
    # Generate MALICIOUS file characteristics
    for i in range(num_samples // 2):
        features = {
            'file_name': f'malware_{i}.exe',
            'file_size': np.random.randint(50000, 2000000),
            'entropy': np.random.uniform(6.5, 7.9),  # Higher entropy
            'num_sections': np.random.randint(4, 8),
            'has_pe_signature': 1,
            'num_strings': np.random.randint(100, 500),
            'avg_string_length': np.random.uniform(6, 15),
            'has_VirtualAlloc': np.random.choice([0, 1], p=[0.3, 0.7]),
            'has_VirtualProtect': np.random.choice([0, 1], p=[0.4, 0.6]),
            'has_CreateRemoteThread': np.random.choice([0, 1], p=[0.6, 0.4]),
            'has_WriteProcessMemory': np.random.choice([0, 1], p=[0.5, 0.5]),
            'total_suspicious_apis': np.random.randint(1, 4),
            'label': 1  # Malicious
        }
        data.append(features)
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset saved to {output_path}")
    print(f"   - Benign samples: {sum(df['label']==0)}")
    print(f"   - Malicious samples: {sum(df['label']==1)}")
    
    return df

def train_model(data_path='data/features.csv', model_path='models/rf_model.pkl'):
    """
    Train Random Forest classifier
    Returns accuracy score
    """
    # Load data
    if not os.path.exists(data_path):
        print("Dataset not found. Generating synthetic data...")
        df = generate_synthetic_dataset()
    else:
        df = pd.read_csv(data_path)
    
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Benign: {sum(df['label']==0)}, Malicious: {sum(df['label']==1)}")
    
    # Prepare features
    X = df.drop(['label', 'file_name'], axis=1, errors='ignore')
    y = df['label']
    feature_columns = X.columns.tolist()
    
    print(f"\nFeatures used: {len(feature_columns)}")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i}. {col}")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Training complete!")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"            Benign  Malicious")
    print(f"Actual")
    print(f"Benign      {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Malicious   {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump((model, feature_columns), f)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"{'='*60}\n")
    
    return accuracy

if __name__ == "__main__":
    print("\n🤖 MALWARE ANALYSIS - MODEL TRAINER\n")
    
    # Train the model
    accuracy = train_model()
    
    print(f"\n✅ Setup complete! Model accuracy: {accuracy:.2%}")
    print("\nYou can now run the Flask app with: python app.py")