import pandas as pd
import numpy as np

def replace_missing_values_and_outliers(data):
    columns = [
               "TCP DL Retrans. Vol (Bytes)",
               "TCP UL Retrans. Vol (Bytes)",
               "Avg RTT DL (ms)",
               "Avg RTT UL (ms)",
               "Avg Bearer TP DL (kbps)",
               "Avg Bearer TP UL (kbps)" ]
    for column in columns:
        mean_value = data[column].mean()
        std_value = data[column].std()
        # Avoid using inplace parameter with chained assignment
        data[column] = data[column].fillna(mean_value)
        data[column] = np.where(
            data[column] > mean_value + 3 * std_value,
            mean_value, data[column]
        )

def aggregate_per_customer(data):
    data['avg_tcp_retransmission'] = data["TCP DL Retrans. Vol (Bytes)"] + data["TCP UL Retrans. Vol (Bytes)"]
    data['avg_rtt'] = data["Avg RTT DL (ms)"] + data["Avg RTT UL (ms)"]
    data['avg_throughput'] = data["Avg Bearer TP DL (kbps)"] + data["Avg Bearer TP UL (kbps)"]

    agg_data = data.groupby('IMSI').agg({
        'avg_tcp_retransmission': 'mean',
        'avg_rtt': 'mean',
        'Handset Type': 'first',
        'avg_throughput': 'mean'
    }).reset_index()

    return agg_data

def get_top_values(data, column, n=10):
    return data.nlargest(n, column)

def get_bottom_values(data, column, n=10):
    return data.nsmallest(n, column)

def get_most_frequent_values(data, column, n=10):
    return data[column].value_counts().nlargest(n).reset_index()

def distribution_per_handset_type(data, column):
    return data.groupby('Handset Type')[column].mean().reset_index()

def extract_experience_metrics(data):
    data['avg_tcp_retransmission'] = data["TCP DL Retrans. Vol (Bytes)"] + data["TCP UL Retrans. Vol (Bytes)"]
    data['avg_rtt'] = data["Avg RTT DL (ms)"] + data["Avg RTT UL (ms)"]
    data['avg_throughput'] = data["Avg Bearer TP DL (kbps)"] + data["Avg Bearer TP UL (kbps)"]

    return data.groupby('IMSI').agg({
        'avg_tcp_retransmission': 'mean',
        'avg_rtt': 'mean',
        'avg_throughput': 'mean'
    }).reset_index()
