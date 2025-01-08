import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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