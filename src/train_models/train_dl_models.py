import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

def train_deep_models(X_train, X_test, y_train, y_test, models_dir='models'):
    """
    Trainiert und speichert DNN, CNN, SimpleRNN, LSTM sowie einen Autoencoder auf den übergebenen Daten.
    """
    os.makedirs(models_dir, exist_ok=True)
    num_classes = len(np.unique(y_train))
    # One-hot Kodierung
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    input_dim = X_train.shape[1]
    X_train_seq = X_train.reshape(-1, input_dim, 1)
    X_test_seq = X_test.reshape(-1, input_dim, 1)

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # DNN
    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    dnn.save(os.path.join(models_dir, 'dnn_model.h5'))

    # CNN
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

    # SimpleRNN
    srnn = Sequential([
        SimpleRNN(64, input_shape=(input_dim, 1)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    srnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    srnn.fit(X_train_seq, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    srnn.save(os.path.join(models_dir, 'srnn_model.h5'))

    # LSTM
    lstm = Sequential([
        LSTM(64, input_shape=(input_dim, 1)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_seq, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    lstm.save(os.path.join(models_dir, 'lstm_model.h5'))

    # Autoencoder
    benign_idx = np.where(y_train == 0)[0]
    X_benign = X_train[benign_idx]
    ae_input = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(ae_input)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(ae_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X_benign, X_benign, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)
    # Speichere im nativen Keras-Format
    autoencoder.save(os.path.join(models_dir, 'ae_model.keras'))
    # Threshold speichern
    recon_train = np.mean((autoencoder.predict(X_benign) - X_benign)**2, axis=1)
    threshold = np.percentile(recon_train, 95)
    np.save(os.path.join(models_dir, 'ae_threshold.npy'), threshold)

    print("Deep Learning Modelle und Autoencoder wurden trainiert und gespeichert.")