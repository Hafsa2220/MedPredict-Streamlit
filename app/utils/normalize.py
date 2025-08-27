# app/utils/normalize.py
import re

def normalize_columns(df):
    # mapping -> canonique -> variantes possibles (ajoute si besoin)
    colmap = {
        "timestamp":       ["timestamp", "date", "horodatage", "time", "datetime"],
        "equipment_name":  ["equipment_name", "equipment", "nom_équipement", "nom equipement", "équipement"],
        "module":          ["module", "module_concerné", "module concerne", "module_id", "module name"],
        "message":         ["message", "description", "message_alarme", "msg", "alarm_message"],
        "duration_h":      ["durée(h)", "duree(h)", "duration", "duree", "durée"],
        "severity":        ["severity", "criticité", "criticite"],
        "event_id":        ["id_événement", "id_evenement", "event_id", "event"],
        "anomaly_score":   ["anomaly_score", "score", "score_anomalie"],
    }
    # index par lower pour faire la correspondance robuste
    lower_to_real = {c.lower(): c for c in df.columns}
    for canon, variants in colmap.items():
        for v in variants:
            vv = v.lower()
            if vv in lower_to_real:
                df.rename(columns={lower_to_real[vv]: canon}, inplace=True)
                break
    return df
