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
    Verknüpft Suricata-Ergebnisse mit den wahren Labels aus der CSV anhand
    von Source IP, Source Port, Destination IP und Destination Port und berechnet
    Accuracy, Precision, Recall und F1-Score.
    """
    # Lade Suricata-Ereignisse
    df_events = load_suricata_events(eve_path)
    # NUR Events, die Flows abbilden (z.B. event_type 'flow', 'alert' oder 'anomaly')
    df_alerts = df_events[df_events['event_type'].isin(['flow', 'alert', 'anomaly'])].copy()

    # Lade Ground-Truth-Daten
    df = pd.read_csv(csv_path)
    # Spaltennamen säubern
    df.columns = df.columns.str.strip()

    # True Labels binär codieren: BENIGN -> 0, alles andere -> 1
    if 'Label' not in df.columns:
        raise KeyError("Die CSV muss eine Spalte 'Label' enthalten.")
    df['true_label'] = df['Label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)

    # 4-Tupel Matching: src_ip, src_port, dest_ip, dest_port
    # In Suricata-Daten
    df_alerts['tuple'] = list(zip(
        df_alerts['src_ip'], df_alerts['src_port'],
        df_alerts['dest_ip'], df_alerts['dest_port']
    ))
    detected_tuples = set(df_alerts['tuple'])

    # In CSV-Daten
    required = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port']
    if not all(col in df.columns for col in required):
        raise KeyError("Zum Abgleich fehlen in der CSV die Spalten Source/Destination IP und Ports.")
    df['tuple'] = list(zip(
        df['Source IP'], df['Source Port'],
        df['Destination IP'], df['Destination Port']
    ))

    # Vorhersage: 1, wenn Flow in Suricata erkannt wurde, sonst 0
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
    print("Dateien 'suricata_metrics.csv' und 'suricata_classification_report.csv' unter data/output/suricata gespeichert.")

    # Lade den Bericht
    df_report = pd.read_csv('data/output/suricata/suricata_classification_report.csv', index_col=0)

    # Wähle die Zeile 'weighted avg'
    row = df_report.loc['weighted avg']

    # Metriken auswählen
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics['Accuracy'],
        row['precision'],
        row['recall'],
        row['f1-score']
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics_to_plot, values, width=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Suricata Performance (weighted avg)')
    ax.grid(True, axis='y')

    # Prozentwerte unter den Balken
    for bar, val in zip(bars, values):
        cx = bar.get_x() + bar.get_width()/2
        ax.text(cx, 0, f"{val:.2f}", ha='center', va='top', fontsize=10)

    plt.tight_layout()
    plt.savefig('data/output/suricata/suricata_bar_chart.png')
    plt.show()
