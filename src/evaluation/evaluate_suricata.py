import json
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_suricata_events(eve_path):
    """
    Liest die Suricata eve.json Zeile für Zeile ein und gibt einen DataFrame zurück.
    Öffnet die Datei mit UTF-8 Encoding und ignoriert Decodierungsfehler.
    """
    events = []
    with open(eve_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(events)


def evaluate_suricata(eve_path: str, csv_path: str):
    """
    Verknüpft Suricata-Ergebnisse mit den wahren Labels aus der CSV und berechnet Accuracy, Precision, Recall und F1-Score.
    Unterstützt Abgleich über 'flow_id' oder über das Fünf-Tupel (src_ip, src_port, dest_ip, dest_port, proto).
    """
    # Lade Suricata-Ereignisse
    df_events = load_suricata_events(eve_path)
    df_alerts = df_events[df_events['event_type'].isin(['alert', 'anomaly'])].copy()

    # Lade Ground-Truth-Daten
    df = pd.read_csv(csv_path)
    # Spaltennamen säubern
    df.columns = df.columns.str.strip()

    # True Labels binär codieren: BENIGN -> 0, alles andere -> 1
    if 'Label' not in df.columns:
        raise KeyError("Die CSV muss eine Spalte 'Label' enthalten.")
    df['true_label'] = df['Label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)

    # Suricata-Vorhersagen erzeugen
    if 'flow_id' in df.columns and 'flow_id' in df_alerts.columns:
        # Direkter Abgleich über flow_id
        detected_flows = df_alerts['flow_id'].unique()
        df['suricata_pred'] = df['flow_id'].isin(detected_flows).astype(int)
    else:
        # Abgleich über Fünf-Tupel
        required = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
        if not all(col in df.columns for col in required):
            raise KeyError("Zum Abgleich fehlen in der CSV die Spalten für Source/Destination IP, Ports und Protocol.")
        # Tupel in Suricata-Daten erstellen
        df_alerts['tuple'] = list(zip(
            df_alerts['src_ip'], df_alerts['src_port'],
            df_alerts['dest_ip'], df_alerts['dest_port'],
            df_alerts['proto']
        ))
        detected_tuples = set(df_alerts['tuple'])
        # Tupel in CSV-Daten erstellen
        df['tuple'] = list(zip(
            df['Source IP'], df['Source Port'],
            df['Destination IP'], df['Destination Port'],
            df['Protocol']
        ))
        df['suricata_pred'] = df['tuple'].isin(detected_tuples).astype(int)

    # Extrahiere y_true und y_pred
    y_true = df['true_label']
    y_pred = df['suricata_pred']

    # Berechne Metriken
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
    }

    # Ausgabe der Metriken
    print("Suricata Evaluation Ergebnisse:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    # Detaillierten Klassifikationsbericht speichern
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Malicious'], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv('data/output/suricata/suricata_classification_report.csv')

    # Gesamtmetriken speichern
    pd.DataFrame([metrics]).to_csv('data/output/suricata/suricata_metrics.csv', index=False)
    print("Dateien 'suricata_metrics.csv' und 'suricata_classification_report.csv' unter data/suricata/output gespeichert.")

    def plot_bar_from_classification_report():
        # Lade den Bericht
        df_report = pd.read_csv('data/output/suricata/suricata_classification_report.csv', index_col=0)

        # Wähle die Zeile 'weighted avg' oder 'Malicious'
        row = df_report.loc['weighted avg']  # Alternativ: 'Malicious'

        # Metriken auswählen
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1-score']
        values = [
            metrics['Accuracy'],               # aus dem Dictionary mitberechnet
            row['precision'],
            row['recall'],
            row['f1-score']
        ]

        # Balkendiagramm mit Beschriftung
        plt.figure(figsize=(8, 5))
        bars = plt.bar(metrics_to_plot, values, color=['steelblue', 'orange', 'green', 'crimson'])
        plt.ylim(0, 1)
        plt.title('Suricata Classification Report (weighted avg)')
        plt.ylabel('Score')
        plt.grid(True, axis='y')

        # Werte als Text über die Balken
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('data/output/suricata/suricata_bar_chart_from_report.png')
        plt.close()

        plot_bar_from_classification_report()

