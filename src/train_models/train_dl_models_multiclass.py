import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def train_deep_models(X_train, X_test, y_train, y_test, models_dir='models'):
    """
    Trainiert und speichert DNN, CNN, RNN (LSTM) sowie einen Autoencoder auf den übergebenen Daten.
    """
    os.makedirs(models_dir, exist_ok=True)
    num_classes = len(np.unique(y_train))
    # One-hot Kodierung für Klassifikation
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    input_dim = X_train.shape[1]
    # Für CNN/RNN reshape: (samples, timesteps, features)
    X_train_seq = X_train.reshape(-1, input_dim, 1)
    X_test_seq = X_test.reshape(-1, input_dim, 1)

    # EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 1) DNN
    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    dnn.save(os.path.join(models_dir, 'dnn_model.h5'))

    # 2) CNN (1D)
    cnn = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train_seq, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    cnn.save(os.path.join(models_dir, 'cnn_model.h5'))

    # 3) RNN (LSTM)
    rnn = Sequential([
        LSTM(64, input_shape=(input_dim, 1)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    rnn.fit(X_train_seq, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    rnn.save(os.path.join(models_dir, 'rnn_model.h5'))

    # 4) Autoencoder (für Anomalie-Erkennung)
    # Hier: Training auf den benignen Flow-Features (Label == 0)
    benign_idx = np.where(y_train == 0)[0]
    X_benign = X_train[benign_idx]
    ae_input = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(ae_input)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(ae_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_benign, X_benign, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    autoencoder.save(os.path.join(models_dir, 'ae_model.h5'))
    # Schwellenwert für Reconstruction-Error bestimmen (95. Perzentil)
    recon_train = np.mean((autoencoder.predict(X_benign) - X_benign) ** 2, axis=1)
    threshold = np.percentile(recon_train, 95)
    np.save(os.path.join(models_dir, 'ae_threshold.npy'), threshold)

    print("Deep Learning Modelle und Autoencoder wurden trainiert und gespeichert.")


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
