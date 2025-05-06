import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--pct_anomalies', default=0.01, type=float)
args = parser.parse_args()
pct_anomalies = args.pct_anomalies

# Define column names based on UNSW-NB15 feature documentation
col_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
    'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'attack_cat', 'label'
]

# Load raw dataset (no header)
df = pd.read_csv('./data/UNSW/UNSW-NB15_1.csv', header=None, names=col_names)


# Drop irrelevant columns
drop_cols = ['srcip', 'dstip', 'sport', 'dsport', 'Stime', 'Ltime', 'attack_cat']
df = df.drop(columns=drop_cols)

# Convert label to integer
df['label'] = df['label'].astype(int)

# Separate normal and anomalous data
normal_data = df[df['label'] == 0]
anomaly_data = df[df['label'] == 1]

# ----------------------------
# Build training set: 100% normal
# ----------------------------
train_frac = 0.75  # 75% of normal data for training
train_normal = normal_data.sample(frac=train_frac, random_state=42)
test_normal = normal_data.drop(train_normal.index)

# ----------------------------
# Build test set: 50% normal, 50% anomalies
# ----------------------------
num_test_normals = len(test_normal)
num_test_anomalies = min(len(anomaly_data), num_test_normals)
test_anomalies = anomaly_data.sample(n=num_test_anomalies, random_state=42)

test_data = pd.concat([test_normal, test_anomalies]).sample(frac=1, random_state=42).reset_index(drop=True)

# Combine features and labels
train_data = train_normal.copy()

# One-hot encode categorical features
categorical_cols = train_data.select_dtypes(include='object').columns.tolist()
train_data_encoded = pd.get_dummies(train_data, columns=categorical_cols)
test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols)

# Align columns between train and test
train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1, fill_value=0)

# Separate X and y
x_train = train_data_encoded.drop(columns=['label'])
y_train = train_data_encoded['label']
x_test = test_data_encoded.drop(columns=['label'])
y_test = test_data_encoded['label']

# Label encode y (binary)
le = LabelEncoder()
le.fit([0, 1])  # Explicitly fit on both normal and anomaly labels
y_train = le.transform(y_train)
y_test = le.transform(y_test)


# Save the preprocessed dataset
preprocessed_data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test,
    'le': le
}

with open('preprocessed_unsw_data.pkl', 'wb') as f:
    pickle.dump(preprocessed_data, f)
