import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

def evaluate_ml_models_multiclass(X_test, y_test):
    models = ['Random Forest', 'KNN', 'Decision Tree', 'Logistic Regression', 'Naive Bayes']
    overall_metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    def evaluate_model(model_name, model_file):
        print(f"Evaluating {model_name}...")
        model = joblib.load(model_file)
        predictions = model.predict(X_test)

        # Overall metrics
        acc  = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
        rec  = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1   = f1_score(y_test, predictions, average='weighted', zero_division=0)

        # Detailed classification report
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        detailed_report = pd.DataFrame(report).transpose()
        detailed_report.to_csv(f'data/output/ml/{model_name}_classification_report.csv')
        print(f"Detailed report saved for {model_name}.")

        return acc, prec, rec, f1, report

    # Evaluate each model
    metrics = [
        ('Random Forest', 'models/random_forest_model.pkl'),
        ('KNN', 'models/knn_model.pkl'),
        ('Decision Tree', 'models/decision_tree_model.pkl'),
        ('Logistic Regression', 'models/logistic_regression_model.pkl'),
        ('Naive Bayes', 'models/naive_bayes_model.pkl'),
    ]

    for name, path in metrics:
        acc, prec, rec, f1, report = evaluate_model(name, path)
        overall_metrics['Model'].append(name)
        overall_metrics['Accuracy'].append(acc * 100)  # Convert to percentage
        overall_metrics['Precision'].append(prec * 100)
        overall_metrics['Recall'].append(rec * 100)
        overall_metrics['F1-Score'].append(f1 * 100)

    # Save overall results
    overall_df = pd.DataFrame(overall_metrics)
    overall_df.to_csv('data/output/ml/model_metrics.csv', index=False)
    print("Overall metrics saved in 'data/output/ml/model_metrics.csv'")

    # Plot the metrics per model
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    index = np.arange(len(models))

    accuracy = overall_metrics['Accuracy']
    precision = overall_metrics['Precision']
    recall = overall_metrics['Recall']
    f1_scores = overall_metrics['F1-Score']

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
    plt.title('Model Performance Comparison (Overall Metrics)')
    plt.xticks(index + 1.5 * bar_width, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/output/ml/overall_metrics_comparison.png')
    plt.show()

