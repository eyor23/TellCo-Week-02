import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def aggregate_metrics(data):
    """Aggregate engagement metrics per customer"""
    aggregated_data = data.groupby('user').agg({
        'Bearer Id': 'count',           # Sessions frequency
        'Dur. (ms)': 'sum',             # Duration of sessions
        'Total DL (Bytes)': 'sum',      # Total download traffic
        'Total UL (Bytes)': 'sum'       # Total upload traffic
    }).rename(columns={
        'Bearer Id': 'session_frequency',
        'Dur. (ms)': 'total_duration',
        'Total DL (Bytes)': 'total_download',
        'Total UL (Bytes)': 'total_upload'
    })
    return aggregated_data
