import json
import csv
import argparse
from collections import defaultdict, deque

# Map common port numbers to services for the 'service' feature
PORT_TO_SERVICE = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    37: "time",
    53: "domain",
    70: "gopher",
    79: "finger",
    80: "http",
    109: "pop_2",
    110: "pop_3",
    111: "sunrpc",
    113: "auth",
    119: "nntp",
    123: "ntp_u",
    137: "netbios_ns",
    138: "netbios_dgm",
    139: "netbios_ssn",
    143: "imap4",
    179: "bgp",
    443: "http_443",
    515: "printer",
    540: "uucp",  # or "uucp_path"
    543: "klogin",
    544: "kshell",
    993: "imap4",  # IMAPS mapped to imap4
    995: "pop_3",  # POP3S mapped to pop_3
    1433: "sql_net",
    1521: "sql_net",  # Oracle mapped to sql_net
    3306: "sql_net",  # MySQL mapped to sql_net
    3389: "remote_job",
    4000: "http_8001",  # approximate mapping
    8000: "http_8001",
    8001: "http_8001",
    8080: "http_8001",
    2784: "http_2784"
}


class KDDFeatureExtractor:
    """
    Converts Zeek connection logs into KDD-style features using a sliding window
    per source host for statistical context-based feature generation.
    """
    def __init__(self, window_size=100):
        # Maintain a sliding window of connections per source IP
        self.window_size = window_size
        self.host_windows = defaultdict(lambda: deque(maxlen=self.window_size))

    def map_raw(self, raw):
        """
        Convert a single Zeek log entry into a basic KDD-like record.
        """
        rec = {}

        # Basic connection attributes
        rec['src_ip'] = raw.get('id.orig_h')
        rec['dst_ip'] = raw.get('id.resp_h')
        rec['src_port'] = raw.get('id.orig_p')
        rec['dst_port'] = raw.get('id.resp_p')
        rec['duration'] = raw.get('duration', 0.0)

        # Protocol
        proto = raw.get('proto', 'tcp').lower()
        rec['protocol_type'] = proto

        # Service: Prefer Zeek's field, else guess based on port
        rec['service'] = raw.get('service') or PORT_TO_SERVICE.get(rec['dst_port'], 'other')

        # Connection state -> KDD flag
        state = raw.get('conn_state', 'OTH')
        rec['flag'] = state

        # Byte transfer info
        rec['src_bytes'] = raw.get('orig_bytes', raw.get('orig_ip_bytes', 0))
        rec['dst_bytes'] = raw.get('resp_bytes', raw.get('resp_ip_bytes', 0))

        # LAND attack indicator: 1 if source IP and port equals destination
        rec['land'] = int(rec['src_ip'] == rec['dst_ip'] and rec['src_port'] == rec['dst_port'])

        # Placeholder fields (not extractable from Zeek)
        rec['wrong_fragment'] = raw.get('weird_fragment_count', 0)
        rec['urgent'] = raw.get('tcp_flags_urg', 0)

        # Static values for unsupported content features
        for field in [
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login'
        ]:
            rec[field] = 0

        return rec

    def compute_window_stats(self, rec):
        """
        Computes statistical features over a sliding window of recent connections
        originating from the same source IP as the current record.
        """
        host = rec['src_ip']
        window = self.host_windows[host]
        stats = {}

        # General connection statistics
        stats['count'] = len(window)
        stats['srv_count'] = sum(1 for r in window if r['service'] == rec['service'])

        # Error rates
        s0 = sum(1 for r in window if r['flag'] == 'S0')
        rej = sum(1 for r in window if r['flag'] == 'REJ')
        stats['serror_rate'] = s0 / stats['count'] if stats['count'] else 0
        stats['srv_serror_rate'] = sum(
            1 for r in window if r['flag'] == 'S0' and r['service'] == rec['service']
        ) / stats['srv_count'] if stats['srv_count'] else 0
        stats['rerror_rate'] = rej / stats['count'] if stats['count'] else 0
        stats['srv_rerror_rate'] = sum(
            1 for r in window if r['flag'] == 'REJ' and r['service'] == rec['service']
        ) / stats['srv_count'] if stats['srv_count'] else 0

        # Service distribution
        stats['same_srv_rate'] = stats['srv_count'] / stats['count'] if stats['count'] else 0
        stats['diff_srv_rate'] = (stats['count'] - stats['srv_count']) / stats['count'] if stats['count'] else 0

        # Host/service diversity
        unique_hosts = len({r['dst_ip'] for r in window})
        stats['srv_diff_host_rate'] = (
            (stats['srv_count'] - sum(1 for r in window if r['dst_ip'] == rec['dst_ip'] and r['service'] == rec['service']))
            / stats['srv_count']
        ) if stats['srv_count'] else 0

        # Destination host specific stats
        dst_window = [r for r in window if r['dst_ip'] == rec['dst_ip']]
        stats['dst_host_count'] = len(dst_window)
        stats['dst_host_srv_count'] = sum(1 for r in dst_window if r['service'] == rec['service'])

        stats['dst_host_same_srv_rate'] = (
            stats['dst_host_srv_count'] / stats['dst_host_count']
        ) if stats['dst_host_count'] else 0

        stats['dst_host_diff_srv_rate'] = (
            (stats['dst_host_count'] - stats['dst_host_srv_count']) / stats['dst_host_count']
        ) if stats['dst_host_count'] else 0

        stats['dst_host_same_src_port_rate'] = (
            sum(1 for r in dst_window if r['src_port'] == rec['src_port']) / stats['dst_host_count']
        ) if stats['dst_host_count'] else 0

        stats['dst_host_srv_diff_host_rate'] = (
            (stats['dst_host_srv_count'] - sum(
                1 for r in dst_window if r['dst_ip'] == r['dst_ip'] and r['service'] == rec['service']
            )) / stats['dst_host_srv_count']
        ) if stats['dst_host_srv_count'] else 0

        stats['dst_host_serror_rate'] = (
            sum(1 for r in dst_window if r['flag'] == 'S0') / stats['dst_host_count']
        ) if stats['dst_host_count'] else 0

        stats['dst_host_srv_serror_rate'] = (
            sum(1 for r in dst_window if r['flag'] == 'S0' and r['service'] == rec['service']) / stats['dst_host_srv_count']
        ) if stats['dst_host_srv_count'] else 0

        stats['dst_host_rerror_rate'] = (
            sum(1 for r in dst_window if r['flag'] == 'REJ') / stats['dst_host_count']
        ) if stats['dst_host_count'] else 0

        stats['dst_host_srv_rerror_rate'] = (
            sum(1 for r in dst_window if r['flag'] == 'REJ' and r['service'] == rec['service']) / stats['dst_host_srv_count']
        ) if stats['dst_host_srv_count'] else 0

        return stats

    def extract_features(self, raw):
        """
        Converts a raw Zeek connection record into a full feature vector including window-based stats.
        """
        rec = self.map_raw(raw)
        stats = self.compute_window_stats(rec)

        # Add the current record to its host window after computing stats
        self.host_windows[rec['src_ip']].append(rec)

        # Combine raw and statistical features
        features = {**rec, **stats}
        features['label'] = ''  # Label placeholder for downstream usage
        return features
