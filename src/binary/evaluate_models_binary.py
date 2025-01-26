import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate_models_binary(X_test, y_test):
    models = ['Random Forest', 'KNN', 'Decision Tree', 'Logistic Regression', 'Naive Bayes']
    accuracy = []
    precision = []
    recall = []
    f1_scores = []

    def evaluate_model(model_name, model_file):
        print(f"Evaluating {model_name}...")
        model = joblib.load(model_file)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, pos_label='anomaly', zero_division=0)
        rec = recall_score(y_test, predictions, pos_label='anomaly', zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label='anomaly', zero_division=0)
        print(f"{model_name} - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1-Score: {f1}")
        return acc, prec, rec, f1

    # Evaluate each model
    metrics = [
        ('Random Forest', 'models/random_forest_model.pkl'),
        ('KNN', 'models/knn_model.pkl'),
        ('Decision Tree', 'models/decision_tree_model.pkl'),
        ('Logistic Regression', 'models/logistic_regression_model.pkl'),
        ('Naive Bayes', 'models/naive_bayes_model.pkl'),
    ]

    for name, path in metrics:
        acc, prec, rec, f1 = evaluate_model(name, path)
        accuracy.append(acc * 100)  # Convert to percentage
        precision.append(prec * 100)
        recall.append(rec * 100)
        f1_scores.append(f1 * 100)

    # Save results
    predictions_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_scores
    })
    predictions_df.to_csv('data/output/binary_metrics.csv', index=False)
    print("Metrics saved in 'data/output/binary_metrics.csv'")

    # Plot the metrics
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    index = np.arange(len(models))

    bars1 = plt.bar(index, accuracy, bar_width, label='Accuracy', color='blue')
    bars2 = plt.bar(index + bar_width, precision, bar_width, label='Precision', color='green')
    bars3 = plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall', color='orange')
    bars4 = plt.bar(index + 3 * bar_width, f1_scores, bar_width, label='F1-Score', color='red')

    # Add percentage labels above the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}%', ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.title('Model Performance (Binary Classification)')
    plt.xticks(index + 1.5 * bar_width, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/output/binary_metrics_comparison.png')
    plt.show()
