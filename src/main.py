from evaluation.evaluate_dl_models import evaluate_deep_models
from evaluation.evaluate_ml_models_binary import evaluate_ml_models_binary
from src.evaluation.evaluate_ml_models_multiclass import evaluate_ml_models_multiclass
from src.evaluation.evaluate_suricata import evaluate_suricata
from src.train_models.train_dl_models import train_deep_models
from src.train_models.train_ml_models_multiclass import train_ml_model_multiclass
from src.utils.utils import get_data_multiclass
from train_models.train_ml_models_binary import train_ml_models_binary

def main():

    # Load data for traditional ML models
    print("Loading data for models...")
    #X_train_ml, X_test_ml, y_train_ml, y_test_ml = get_data()
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = get_data_multiclass()

    # Train and evaluate traditional ML models
    print("Training ML models...")
    train_ml_models_binary(X_train_ml, y_train_ml)
    train_ml_model_multiclass(X_train_ml, y_train_ml)
    print("Evaluating ML models...")
    evaluate_ml_models_binary(X_test_ml, y_test_ml)
    evaluate_ml_models_multiclass(X_test_ml, y_test_ml)

    # Train und Eval tiefenlernende Modelle
    print("Training Deep Learning Modelle...")
    train_deep_models(X_train_ml, X_test_ml, y_train_ml, y_test_ml)
    print("Evaluating Deep Learning Modelle...")
    evaluate_deep_models(X_test_ml, y_test_ml)

    print("Evaluating Suricata against ground truth…")
    evaluate_suricata(
    eve_path='data/input/suricata_logs/Wednesday_eve.json',
    csv_path='data/input/CIC-IDS/flow_data/Wednesday-WorkingHours_flow_data.csv'
    )

if __name__ == "__main__":
    main()
