import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def evaluate_deep_models(X_test, y_test, models_dir='models'):
    """
    Lädt die Deep-Lernmodelle (DNN, CNN, SimpleRNN, LSTM, Autoencoder),
    berechnet Accuracy, Precision, Recall und F1-Score,
    speichert die Metriken, und zeigt ein Balkendiagramm mit Prozentwerten an.
    """
    results = []
    num_classes = len(np.unique(y_test))

    # Sequenzdaten für RNN/CNN
    X_test_seq = X_test.reshape(-1, X_test.shape[1], 1)
    # One-hot für Multiclass
    y_test_cat = to_categorical(y_test, num_classes)

    # Hilfsfunktion zur Evaluation der Multiclass-Modelle
    def eval_model(name, filename, data, true_cat):
        model = load_model(os.path.join(models_dir, filename), compile=False)
        preds = model.predict(data)
        pred_cls = np.argmax(preds, axis=1)
        true_cls = np.argmax(true_cat, axis=1)
        acc = accuracy_score(true_cls, pred_cls)
        prec = precision_score(true_cls, pred_cls, average='weighted', zero_division=0)
        rec = recall_score(true_cls, pred_cls, average='weighted', zero_division=0)
        f1 = f1_score(true_cls, pred_cls, average='weighted', zero_division=0)
        # detaillierter Report
        report = classification_report(true_cls, pred_cls, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(f'data/output/dl/{name}_classification_report.csv')
        return acc*100, prec*100, rec*100, f1*100

    # DNN
    acc, prec, rec, f1 = eval_model('DNN', 'dnn_model.h5', X_test, y_test_cat)
    results.append({'Model': 'DNN', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    # CNN
    acc, prec, rec, f1 = eval_model('CNN', 'cnn_model.h5', X_test_seq, y_test_cat)
    results.append({'Model': 'CNN', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    # SimpleRNN
    acc, prec, rec, f1 = eval_model('SimpleRNN', 'srnn_model.h5', X_test_seq, y_test_cat)
    results.append({'Model': 'SimpleRNN', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    # LSTM
    acc, prec, rec, f1 = eval_model('LSTM', 'lstm_model.h5', X_test_seq, y_test_cat)
    results.append({'Model': 'LSTM', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})

    # Autoencoder (Binary Anomaly Detection)
    ae = load_model(os.path.join(models_dir, 'ae_model.keras'), compile=False)
    threshold = np.load(os.path.join(models_dir, 'ae_threshold.npy'))
    y_true_bin = (y_test > 0).astype(int)
    recon_error = np.mean((ae.predict(X_test) - X_test) ** 2, axis=1)
    y_pred_bin = (recon_error > threshold).astype(int)
    acc_ae = accuracy_score(y_true_bin, y_pred_bin)
    prec_ae = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec_ae = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1_ae = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    report_ae = classification_report(y_true_bin, y_pred_bin, target_names=['Benign','Malicious'], output_dict=True, zero_division=0)
    pd.DataFrame(report_ae).transpose().to_csv('data/output/dl/AE_classification_report.csv')
    results.append({'Model': 'Autoencoder', 'Accuracy': acc_ae*100, 'Precision': prec_ae*100, 'Recall': rec_ae*100, 'F1-Score': f1_ae*100})

    # Ergebnisse speichern
    df_res = pd.DataFrame(results)
    os.makedirs('data/output/dl', exist_ok=True)
    df_res.to_csv('data/output/dl/deep_model_metrics.csv', index=False)
    print("Deep Learning Evaluation abgeschlossen. Metriken unter data/output/dl/deep_model_metrics.csv gespeichert.")

    # Balkendiagramm plotten
    ax = df_res.set_index('Model').plot(kind='bar', figsize=(10, 6))
    plt.title('Deep Learning Model Performance')
    plt.ylabel('Score (%)')
    plt.ylim(0, 100)
    plt.grid(True, axis='y')
    # Prozentwerte auf den Balken
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()
