from src.multiclass.evaluate_models_multiclass import evaluate_models_multiclass
from src.multiclass.train_models_multiclass import train_and_save_model_multiclass
from src.utils.utils import get_data, get_data_multiclass
from src.binary.train_models import train_and_save_models_binary
from src.binary.evaluate_models import evaluate_models_binary


def main():

    # Load data for traditional ML models
    print("Loading data for ML models...")
    #X_train_ml, X_test_ml, y_train_ml, y_test_ml = get_data_multiclass()
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = get_data()

    # Train and evaluate traditional ML models
    print("Training ML models...")
    #train_and_save_model_multiclass(X_train_ml, y_train_ml)
    train_and_save_models_binary(X_train_ml, y_train_ml)

    print("Evaluating ML models...")
    #evaluate_models_multiclass(X_test_ml, y_test_ml)
    evaluate_models_binary(X_test_ml, y_test_ml)
if __name__ == "__main__":
    main()
