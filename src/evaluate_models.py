import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def evaluate_models(X_test, y_test):
    if len(X_test) != len(y_test):
        print("Error: The number of samples in X_test and y_test do not match.")
        print(f"X_test length: {len(X_test)}, y_test length: {len(y_test)}")
        return
    rf_model = joblib.load('models/random_forest_model.pkl')
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_f1 = f1_score(y_test, rf_predictions, pos_label='anomaly')
    print(f"Random Forest - Accuracy: {rf_accuracy}, F1-Score: {rf_f1}")

    # Load and evaluate SVM
    svm_model = joblib.load('models/svm_model.pkl')
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_f1 = f1_score(y_test, svm_predictions, pos_label='anomaly')
    print(f"SVM - Accuracy: {svm_accuracy}, F1-Score: {svm_f1}")

    # Load and evaluate KNN
    knn_model = joblib.load('models/knn_model.pkl')
    knn_predictions = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    knn_f1 = f1_score(y_test, knn_predictions, pos_label='anomaly')
    print(f"KNN - Accuracy: {knn_accuracy}, F1-Score: {knn_f1}")

    # Save results
    predictions_df = pd.DataFrame({
        'Random_Forest': rf_predictions,
        'SVM': svm_predictions,
        'KNN': knn_predictions,
        'True_Label': y_test
    })
    predictions_df.to_csv('data/predictions_with_accuracy_f1.csv', index=False)
    print("Predictions with metrics saved in 'data/predictions_with_accuracy_f1.csv'")
