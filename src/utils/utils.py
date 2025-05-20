from src.data_preprocessing import load_and_preprocess_dataset1, load_and_preprocess_dataset2, \
    load_and_preprocess_dataset3


def get_data():
    train_path = 'data/input/US_LAN.csv'
    return load_and_preprocess_dataset1(train_path)

def get_data_multiclass():
    train_path = 'data/input/CIC-IDS/Thursday-WorkingHours.csv'
    return load_and_preprocess_dataset3(train_path)