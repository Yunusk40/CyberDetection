import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



def train_and_save_model_multiclass(X_train, y_train):
    print(f"y_train shape: {y_train.shape}")  # Debugging to ensure it has the shape (num_samples, num_classes)

    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("Random Forest trained successfully.")

    # K-Nearest Neighbors
    print("Training KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, 'models/knn_model.pkl')
    print("KNN trained successfully.")

    # Decision Tree
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, 'models/decision_tree_model.pkl')
    print("Decision Tree trained successfully.")

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    print("Logistic Regression trained successfully.")

    # Naive Bayes
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
    print("Naive Bayes trained successfully.")

    print("All models trained and saved successfully.")
