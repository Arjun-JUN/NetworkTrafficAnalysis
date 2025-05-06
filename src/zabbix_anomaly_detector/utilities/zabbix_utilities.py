from zabbix_utils import ZabbixAPI
import json


def get_api(config):
    zapi = ZabbixAPI(config['zabbix_url'])
    zapi.login(user=config['zabbix_user'], password=config['zabbix_password'])
    return zapi


def fetch_item_ids(api, host_id, item_keys):
    items = api.item.get(
        output=['itemid', 'key_', 'value_type'],
        hostids=[host_id],
        filter={'key_': item_keys}
    )
    return {item['key_']: {'id': item['itemid'], 'type': int(item['value_type'])} for item in items}


def fetch_history(api, item_map, time_from, time_till):
    history = []
    for key, meta in item_map.items():
        data = api.history.get(
            itemids=meta['id'],
            time_from=time_from,
            time_till=time_till,
            output='extend',
            history=meta['type']
        )
        values = [float(d['value']) for d in data]
        history.append(values)
    return history

import time
from zabbix_utils import Sender
import subprocess

def send_anomaly_score(api, host_id, key, value):
    # Check if item exists
    existing = api.item.get(
        hostids=[host_id],
        filter={"key_": key},
        output=["itemid"]
    )

    if not existing:
        item = api.item.create(
            name="Anomaly Score",
            key_=key,
            hostid=host_id,
            type=2,           # Zabbix trapper
            value_type=0,     # Numeric float
            delay=0
        )
        print("Created trapper item:", item)
    
    # Now send the value using zabbix_sender
    subprocess.run([
        "zabbix_sender",
        "-z", "127.0.0.1",             # Change this if Zabbix server is remote
        "-s", "Zabbix server",         # Hostname as in Zabbix frontend
        "-k", key,
        "-o", str(value)
    ], check=True)
