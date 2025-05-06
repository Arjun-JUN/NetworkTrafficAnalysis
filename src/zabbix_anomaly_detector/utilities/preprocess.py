import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize a label encoder to convert string labels to integers
le = LabelEncoder()

# Function to preprocess a given KDD dataframe
def preprocess_kdd_dataframe(df: pd.DataFrame):
    """
    Preprocesses the KDD dataset:
    - Renames columns
    - One-hot encodes categorical variables
    - Selects numeric and encoded features
    - Encodes labels
    - Splits into train and test sets
    """
    
    # Define column names (from KDD99 dataset spec)
    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]
    df = df[col_names]

    # One-hot encode protocol_type (e.g., 'tcp', 'udp', 'icmp')
    protocols = ['icmp', 'tcp', 'udp']
    for proto in protocols:
        df[f'protocol_type_{proto}'] = (df['protocol_type'] == proto).astype(np.float32)

    # One-hot encode service types
    services = [
        'IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
        'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784',
        'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm',
        'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private',
        'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i',
        'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois'
    ]
    for service in services:
        col_name = service.lower().replace('-', '_').replace('.', '_')
        df[f'service_{col_name}'] = (df['service'] == service).astype(np.float32)

    # One-hot encode TCP flags
    flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    for flag in flags:
        df[f'flag_{flag}'] = (df['flag'] == flag).astype(np.float32)

    # -------------------- Feature Selection --------------------

    # List of continuous/numeric variables
    numeric_vars = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'land', 'logged_in', 'is_host_login', 'is_guest_login'
    ]

    # Identify manually one-hot encoded categorical columns
    dummy_vars = [
        col for col in df.columns
        if col.startswith('protocol_type_') or col.startswith('service_') or col.startswith('flag_')
    ]

    # Combine numeric and dummy features
    X = df[numeric_vars + dummy_vars]

    # Extract labels
    y = df['label'].values

    # -------------------- Sanity Check --------------------

    # Warn if any column still has string values
    for col in X.columns:
        if X[col].apply(lambda x: isinstance(x, str)).any():
            print(f"Column '{col}' contains string values")

    print(f"Final feature space dimensionality: {X.shape[1]}")


    return X
