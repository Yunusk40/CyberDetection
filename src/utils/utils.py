from src.data_preprocessing import load_and_preprocess_data, load_and_preprocess_multiclass_data, \
    load_and_preprocess_webattack_data


def get_data():
    train_path = 'data/input/dataset1_US_LAN.csv'
    return load_and_preprocess_data(train_path)

def get_data_multiclass():
    train_path = 'data/input/dataset3_WebAttacks.csv'
    return load_and_preprocess_webattack_data(train_path)

    #train_path = 'data/input/dataset3_WebAttacks.csv'
    #return load_and_preprocess_multiclass_data(train_path)


