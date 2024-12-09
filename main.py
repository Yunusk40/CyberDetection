from src.utils import get_data
from src.train_models import train_and_save_models
from src.evaluate_models import evaluate_models


def main():
    # Load and preprocess data (splits internally into train and test)
    X_train, X_test, y_train, y_test = get_data()

    # Train and save models
    train_and_save_models(X_train, y_train)

    # Evaluate models on the test data
    evaluate_models(X_test, y_test)


if __name__ == "__main__":
    main()
