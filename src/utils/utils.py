from src.data_preprocessing import load_and_preprocess_data

def get_data():
    train_path = 'data/dataset1_US_LAN.csv'
    return load_and_preprocess_data(train_path)

def get_data_multiclass():
    train_path = 'data/dataset2_UNSW_NB15.csv'
    return load_and_preprocess_data_multiclass(train_path)