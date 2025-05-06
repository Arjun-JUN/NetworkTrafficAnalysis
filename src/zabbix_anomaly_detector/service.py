import time
from main import *
import numpy as np
from sklearn.preprocessing import StandardScaler

api = get_api(config)
host = api.host.get(filter={"host": config["host_name"]}, output=["hostid", "name"])
host_id = host[0]['hostid']
item_map = fetch_item_ids(api, host_id, config['item_keys'])

while True:
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

    scores = get_anomaly_scores(X)
    latest_score = scores[-1][0]
    print(f"[{time.ctime()}] Anomaly Score: {latest_score}")

    send_anomaly_score(api, host_id, 'custom.anomaly.score', latest_score)

    time.sleep(30)  # sleep 5 minutea