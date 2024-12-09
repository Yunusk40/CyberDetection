from src.data_preprocessing import load_and_preprocess_data

def get_data():
    train_path = 'data/Train_data.csv'
    return load_and_preprocess_data(train_path)
