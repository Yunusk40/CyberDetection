from src.multiclass_classification.evaluate_models_multiclass import evaluate_models_multiclass
from src.multiclass_classification.train_models_multiclass import train_and_save_model_multiclass
from src.utils.utils import get_data, get_data_multiclass
from src.binary_classifikation.train_models_binary import train_and_save_models_binary
from src.binary_classifikation.evaluate_models_binary import evaluate_models_binary


def main():

    # Load data for traditional ML models
    print("Loading data for ML models...")
    #X_train_ml, X_test_ml, y_train_ml, y_test_ml = get_data()
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = get_data_multiclass()

    # Train and evaluate traditional ML models
    print("Training ML models...")
    #train_and_save_models_binary(X_train_ml, y_train_ml)
    train_and_save_model_multiclass(X_train_ml, y_train_ml)

    print("Evaluating ML models...")
    #evaluate_models_binary(X_test_ml, y_test_ml)
    evaluate_models_multiclass(X_test_ml, y_test_ml)
if __name__ == "__main__":
    main()
