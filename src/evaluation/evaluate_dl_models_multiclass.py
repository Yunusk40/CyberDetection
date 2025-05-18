import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_deep_models(X_test, y_test, models_dir='models'):
    """
    Lädt die gespeicherten Deep-Modelle, erstellt Vorhersagen und berechnet Metriken.
    Für DNN, CNN, RNN: Multiclass-Classification.
    Für Autoencoder: Binary Anomaly Detection (Benign vs Malicious).
    """
    results = []
    num_classes = len(np.unique(y_test))
    # Prepare test data
    X_test_seq = X_test.reshape(-1, X_test.shape[1], 1)
    # Multiclass One-hot
    y_test_cat = to_categorical(y_test, num_classes)

    # Hilfsfunktion
    def eval_model(name, model_path, data, true_cat=None):
        model = load_model(os.path.join(models_dir, model_path))
        preds = model.predict(data)
        pred_cls = np.argmax(preds, axis=1)
        true_cls = np.argmax(true_cat, axis=1) if true_cat is not None else None
        acc = accuracy_score(true_cls, pred_cls)
        prec = precision_score(true_cls, pred_cls, average='weighted', zero_division=0)
        rec = recall_score(true_cls, pred_cls, average='weighted', zero_division=0)
        f1 = f1_score(true_cls, pred_cls, average='weighted', zero_division=0)
        # Speichern detaillierter Report
        report = classification_report(true_cls, pred_cls, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(f'data/output/{name}_classification_report.csv')
        return acc, prec, rec, f1

    # 1) DNN
    acc, prec, rec, f1 = eval_model('DNN', 'dnn_model.h5', X_test, y_test_cat)
    results.append({'Model': 'DNN', 'Accuracy': acc*100, 'Precision': prec*100, 'Recall': rec*100, 'F1-Score': f1*100})
    # 2) CNN
    acc, prec, rec, f1 = eval_model('CNN', 'cnn_model.h5', X_test_seq, y_test_cat)
    results.append({'Model': 'CNN', 'Accuracy': acc*100, 'Precision': prec*100, 'Recall': rec*100, 'F1-Score': f1*100})
    # 3) RNN
    acc, prec, rec, f1 = eval_model('RNN', 'rnn_model.h5', X_test_seq, y_test_cat)
    results.append({'Model': 'RNN', 'Accuracy': acc*100, 'Precision': prec*100, 'Recall': rec*100, 'F1-Score': f1*100})

    # 4) Autoencoder (Binary)
    ae = load_model(os.path.join(models_dir, 'ae_model.h5'))
    threshold = np.load(os.path.join(models_dir, 'ae_threshold.npy'))
    # Binary true labels: benign=0, malicious=1
    y_true_bin = (y_test > 0).astype(int)
    recon_error = np.mean((ae.predict(X_test) - X_test) ** 2, axis=1)
    y_pred_bin = (recon_error > threshold).astype(int)
    # Metriken
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    # Report speichern
    report = classification_report(y_true_bin, y_pred_bin, target_names=['Benign', 'Malicious'], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv('data/output/AE_classification_report.csv')
    results.append({'Model': 'Autoencoder', 'Accuracy': acc*100, 'Precision': prec*100, 'Recall': rec*100, 'F1-Score': f1*100})

    # Ergebnisse speichern
    df_res = pd.DataFrame(results)
    df_res.to_csv('data/output/deep_model_metrics.csv', index=False)
    print("Deep Learning Evaluation abgeschlossen. Metriken in 'data/output/deep_model_metrics.csv' gespeichert.")