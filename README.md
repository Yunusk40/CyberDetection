This project implements a Machine Learning (ML) pipeline to classify network traffic data into normal or malicious activities. It supports both binary classification (e.g., BENIGN vs. ANOMALY) and multiclass classification (e.g., specific web attacks). The project preprocesses the data, trains various ML models, and evaluates their performance.

Getting Started
Install Required Python libraries:
pip install -r requirements.txt

Datasets
to run the project, download and place the following datasets under the data/input/ directory:

1. Train Data (https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection?resource=download&select=Train_data.csv)
Save as: dataset1_Train_data.csv

2. UNSW-NB15 Training Data (https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data?select=UNSW_NB15_training-set.csv)
Save as: dataset2_UNSW_NB15.csv

3. Thursday Working Hours Web Attacks Data (https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?select=Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv)
Save as: dataset3_WebAttacks.csv

How to Run
1. Place the downloaded datasets in the data/input/ directory.
2. Run the Main Script Execute the main script to train and evaluate models

Results are saved in data/output
