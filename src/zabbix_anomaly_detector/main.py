import json
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from utilities.zabbix_utilities import *        # Functions to interact with Zabbix API
from utilities.model_utilities import *         # Utilities for model loading, scoring, etc.
from utilities.zeek_extractor import *          # Zeek log feature extractor (e.g., KDD99-style)
from utilities.preprocess import *              # Preprocessing functions like preprocess_kdd_dataframe

# ----------------------------- Load Configuration -----------------------------
with open("config.json") as f:
    config = json.load(f)

# Connect to Zabbix API
api = get_api(config)

# Set time range (last hour)
time_till = int(time.time())
time_from = time_till - 3600

# Get host ID from Zabbix using host name in config
host = api.host.get(filter={"host": config["host_name"]}, output=["hostid", "name"])
host_id = host[0]['hostid']
print(f"Host ID: {host_id}")

# ----------------------------- Load Zeek Connection Logs -----------------------------
zeek_log_path = "/usr/local/zeek/logs/current/"
conn_log = zeek_log_path + "conn.log"

# Initialize KDD feature extractor
extractor = KDDFeatureExtractor(window_size=100)

# Extract features from Zeek logs
rows = []
with open(conn_log) as infile:
    reader = (json.loads(line) for line in infile)  # Parse each line as JSON

    for raw in reader:
        try:
            feats = extractor.extract_features(raw)  # Extract KDD-style features
            rows.append(feats)
        except Exception as e:
            print(f"Skipping line due to error: {e}")  # Skip any malformed entries

# Convert extracted features into a DataFrame
df = pd.DataFrame(rows)
print("Extracted DataFrame shape:\t", df.shape)
df_processed = preprocess_kdd_dataframe(df)
print("Preprocessed DataFrame shape:\t",df_processed.shape)

# ----------------------------- Load Model and Predict -----------------------------

X = df_processed.astype(np.float32)

# Compute anomaly scores using model
print(X.shape)
scores = get_anomaly_scores(X)

# ----------------------------- Send Result to Zabbix -----------------------------

# Find and print the row numbers with anomaly detected
for idx, score in enumerate(scores):
    if score != 0 :
        print(f"Anomaly detected at row {idx} with score {score}")

# Send the most recent anomaly score to Zabbix
send_anomaly_score(api, host_id, 'custom.anomaly.score', scores[-1][0])
