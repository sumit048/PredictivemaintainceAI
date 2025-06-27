import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

warnings.filterwarnings("ignore")


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the predictive maintenance dataset
    """
    print("Loading data...")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # Drop unnecessary columns
    df.drop(columns=['UDI', 'Product ID'], inplace=True)

    # Feature Engineering
    print("Creating new features...")
    # Temperature difference
    df['temperature_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

    # Mechanical Power
    df['Mechanical Power [W]'] = np.round(
        (df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi) / 60, 4
    )

    # Drop failure-specific columns (highly correlated with target)
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df.drop(columns=failure_cols, inplace=True)

    return df


def encode_and_scale_features(df):
    """
    Encode categorical variables and scale numerical features
    """
    print("Encoding categorical variables...")

    # Label encode the 'Type' column
    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])

    # Separate features and target
    y = df.pop("Machine failure")
    X = df

    print("Scaling features...")
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    return X_scaled, y, scaler, label_encoder


def handle_imbalanced_data(X, y):
    """
    Handle imbalanced dataset using SMOTE
    """
    print("Handling imbalanced data...")
    print(f"Before SMOTE: {Counter(y)}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"After SMOTE: {Counter(y_resampled)}")

    return X_resampled, y_resampled


def train_model(X, y):
    """
    Train Random Forest model
    """
    print("Training Random Forest model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, accuracy


def save_model_artifacts(model, scaler, label_encoder, feature_names, model_dir="models"):
    """
    Save model and preprocessing artifacts
    """
    print("Saving model artifacts...")

    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    with open(f"{model_dir}/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save label encoder
    with open(f"{model_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Save feature names
    with open(f"{model_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print(f"Model artifacts saved in '{model_dir}' directory")


def train_predictive_maintenance_model(data_path, model_dir="models"):
    """
    Complete training pipeline for predictive maintenance model

    Args:
        data_path (str): Path to the dataset CSV file
        model_dir (str): Directory to save model artifacts

    Returns:
        dict: Training results and model information
    """
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(data_path)

        # Encode and scale features
        X, y, scaler, label_encoder = encode_and_scale_features(df)

        # Handle imbalanced data
        X_resampled, y_resampled = handle_imbalanced_data(X, y)

        # Train model
        model, accuracy = train_model(X_resampled, y_resampled)

        # Save artifacts
        save_model_artifacts(model, scaler, label_encoder, X.columns.tolist(), model_dir)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        results = {
            'success': True,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'model_path': f"{model_dir}/random_forest_model.pkl",
            'message': 'Model trained and saved successfully!'
        }

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Model Accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_dir}/")
        print("\nTop 5 Important Features:")
        print(feature_importance.head())

        return results

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Training failed!'
        }


if __name__ == "__main__":
    # Example usage
    data_path = "ai4i2020.csv"  # Update this path
    results = train_predictive_maintenance_model(data_path)

    if results['success']:
        print(f"\nTraining completed with accuracy: {results['accuracy']:.4f}")
    else:
        print(f"Training failed: {results['message']}")