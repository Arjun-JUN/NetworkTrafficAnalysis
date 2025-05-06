import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
import pickle

# -------------------- Argument Parsing --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--pct_anomalies', default=0.01, type=float,
                    help='Proportion of anomalies to keep relative to number of normal samples')
args = parser.parse_args()
pct_anomalies = args.pct_anomalies

# -------------------- Load Dataset --------------------
data_path = './data/kddcup.data.corrected'

# Define column names as per the KDD dataset documentation
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Load the CSV data
df = pd.read_csv(data_path, header=None, names=col_names, index_col=False)

# Initialize label encoder and fit it on labels
le = LabelEncoder()
le.fit(df['label'])

# -------------------- Anomaly Reduction --------------------
def reduce_anomalies(df, pct_anomalies=0.01):
    """
    Reduce the number of anomalies to be a fraction of normal samples.
    """
    labels = df['label']
    is_anomaly = labels != 'normal.'
    num_normal = np.sum(~is_anomaly)
    num_anomalies = int(pct_anomalies * num_normal)
    
    all_anomalies = df[is_anomaly]
    anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
    anomalous_data = df.loc[anomalies_to_keep]
    normal_data = df[~is_anomaly]
    
    new_df = pd.concat([normal_data, anomalous_data], axis=0)
    return new_df

# Apply anomaly reduction
df = reduce_anomalies(df, pct_anomalies=pct_anomalies)


# -------------------- One-Hot Encoding (Manual) --------------------

# One-hot encode protocol types
protocols = ['icmp', 'tcp', 'udp']
for proto in protocols:
    df[f'protocol_type_{proto}'] = (df['protocol_type'] == proto).astype(np.float32)

# One-hot encode service types
services = [
    'IRC','X11','Z39_50','aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
    'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames','http','http_2784',
    'http_443','http_8001','imap4','iso_tsap','klogin','kshell','ldap','link','login','mtp','name','netbios_dgm',
    'netbios_ns','netbios_ssn','netstat','nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3','printer','private',
    'red_i','remote_job','rje','shell','smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i',
    'time','urh_i','urp_i','uucp','uucp_path','vmnet','whois'
]
for service in services:
    col_name = service.lower().replace('-', '_').replace('.', '_')
    df[f'service_{col_name}'] = (df['service'] == service).astype(np.float32)

# One-hot encode flag types
flags = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
for flag in flags:
    df[f'flag_{flag}'] = (df['flag'] == flag).astype(np.float32)


# -------------------- Feature Selection --------------------

# Numeric features
numeric_vars = [
    'duration','src_bytes','dst_bytes','wrong_fragment','urgent','hot','num_failed_logins',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'land','logged_in','is_host_login','is_guest_login'
]

# Dummy vars (manually one-hot encoded categorical variables)
dummy_vars = [col for col in df.columns if col.startswith('protocol_type_') 
              or col.startswith('service_') or col.startswith('flag_')]

# Final feature matrix
X = df[numeric_vars + dummy_vars]

# -------------------- Sanity Check --------------------

# Print any columns containing string data
for col in X.columns:
    if X[col].apply(lambda x: isinstance(x, str)).any():
        print(f"Column '{col}' contains string values")

print(f"Final feature space dimensionality: {X.shape[1]}")

# -------------------- Label Processing --------------------

# Extract and encode labels
labels = df['label'].copy()
integer_labels = le.transform(labels)

# -------------------- Train-Test Split --------------------

x_train, x_test, y_train, y_test = train_test_split(
    X, integer_labels, test_size=0.25, random_state=42
)

# -------------------- Save Preprocessed Data --------------------

preprocessed_data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test,
    'le': le
}

# Save as a pickle file
path = 'preprocessed_data_full.pkl'
with open(path, 'wb') as out:
    pickle.dump(preprocessed_data, out)
