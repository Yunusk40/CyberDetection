import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def train_and_save_models(X_train, y_train):
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')

    # Support Vector Machine
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, 'models/svm_model.pkl')

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, 'models/knn_model.pkl')

    print("Models trained and saved successfully.")
