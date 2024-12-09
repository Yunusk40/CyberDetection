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
