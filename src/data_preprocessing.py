import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(train_path):
    # Load the full dataset (train and pseudo-test)
    data = pd.read_csv(train_path)

    # Identify non-numeric columns
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    non_numeric_columns = non_numeric_columns.drop('class')  # Remove 'class' from non-numeric columns

    # Apply One-Hot Encoding to non-numeric columns
    data = pd.get_dummies(data, columns=non_numeric_columns)

    # Separate features and target
    X = data.drop(columns=['class'])
    y = data['class']

    # Split into training and pseudo-testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def load_and_preprocess_multiclass_data(dataset_path):
    # Load the full dataset
    data = pd.read_csv(dataset_path)

    # Check for missing values and handle them (if necessary)
    if data.isnull().sum().sum() > 0:
        print("Dataset contains missing values. Filling with mode...")
        data.fillna(data.mode().iloc[0], inplace=True)

    # Identify non-numeric columns (excluding the target columns 'attack_cat' and 'label')
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    non_numeric_columns = non_numeric_columns.drop(['attack_cat'], errors='ignore')  # Keep 'attack_cat' as target

    # Apply One-Hot Encoding to non-numeric columns
    data = pd.get_dummies(data, columns=non_numeric_columns)

    # Separate features and target
    X = data.drop(columns=['attack_cat', 'label'])  # Drop both 'attack_cat' and 'label' from features
    y = data['attack_cat']  # Use 'attack_cat' for multiclassification

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def load_and_preprocess_webattack_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Handle missing values
    if 'Flow Bytes/s' in data.columns:
        data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(data['Flow Bytes/s'].median())
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Only fill missing values for numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Ensure no remaining NaN values
    if data.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains NaN values even after cleaning.")

    # Encode the target variable (Label)
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    # Split data into features and target
    X = data.drop(columns=['Label'])
    y = data['Label']

    # Split into training and testing sets (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Analyze class distribution in the test set
    print("Class distribution in the test set:")
    class_distribution = y_test.value_counts()
    print(class_distribution)

    # Optionally: Convert to percentages for better understanding
    class_distribution_percent = y_test.value_counts(normalize=True) * 100
    print("\nClass distribution in percentages:")
    print(class_distribution_percent)

    # Preprocess training and testing data independently
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test



