This project implements a Machine- and Deep Learning pipeline to classify network traffic data into normal or malicious activities. It supports both binary classification (e.g., BENIGN vs. ANOMALY) and multiclass classification (e.g., specific web attacks). The project preprocesses the data, trains various ML/DL models, and evaluates their performance.

Getting Started
Install Required Python libraries:
pip install -r requirements.txt

Datasets
to run the project, download and place the following datasets under the data/input/ directory:

Download the csv and pcap files from the CICIDS 2017 dataset:
(https://www.unb.ca/cic/datasets/ids-2017.html)
save them in data/input

For the ML/DL evaluation, you just need the csv files from the CICIDS 2017 dataset.

In order to use the suricata evualation, you first have to 

1. download suricata from https://suricata.io/download/ and afterwards run the pcap files with the following command:
2. `suricata.exe -c suricata.yaml -r <pcap_file>`
3. save the log files in data/input/suricata_logs/
4. You also need to use the csv files with the extra flow information from the CICIDS 2017 dataset

**How to Run** 

Run the Main Script Execute the main script to train and evaluate models

Results are saved in data/output
