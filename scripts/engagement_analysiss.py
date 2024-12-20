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

def top_customers(aggregated_data):
    """Report the top 10 customers per engagement metric"""
    top_customers_frequency = aggregated_data.nlargest(10, 'session_frequency')
    top_customers_duration = aggregated_data.nlargest(10, 'total_duration')
    top_customers_download = aggregated_data.nlargest(10, 'total_download')
    top_customers_upload = aggregated_data.nlargest(10, 'total_upload')

    return {
        'frequency': top_customers_frequency,
        'duration': top_customers_duration,
        'download': top_customers_download,
        'upload': top_customers_upload
    }

def normalize_and_cluster(aggregated_data, n_clusters=3):
    """Normalize metrics and run K-Means clustering"""
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(aggregated_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(normalized_data)
    aggregated_data['cluster'] = clusters

    return aggregated_data, scaler, kmeans

def compute_cluster_stats(aggregated_data):
    """Compute statistics for each cluster"""
    cluster_stats = aggregated_data.groupby('cluster').agg({
        'session_frequency': ['min', 'max', 'mean', 'sum'],
        'total_duration': ['min', 'max', 'mean', 'sum'],
        'total_download': ['min', 'max', 'mean', 'sum'],
        'total_upload': ['min', 'max', 'mean', 'sum']
    })
    return cluster_stats

def user_traffic_per_app(data):
    """Aggregate user total traffic per application"""
    application_columns = [
        'Social Media DL (Bytes)',
        'Google DL (Bytes)',
        'Email DL (Bytes)',
        'Youtube DL (Bytes)',
        'Netflix DL (Bytes)',
        'Gaming DL (Bytes)',
        'Other DL (Bytes)'
    ]
    application_data = data.groupby('user')[application_columns].sum()
    return application_data

def top_users_per_application(application_data):
    """Derive the top 10 most engaged users per application"""
    top_users_per_app = {}
    for app in application_data.columns:
        top_users_per_app[app] = application_data.nlargest(10, app)
    return top_users_per_app

def plot_top_apps(application_data):
    """Plot the top 3 most used applications"""
    top_3_apps = application_data.sum().nlargest(3).index
    top_3_apps_data = application_data[top_3_apps]

    top_3_apps_data.plot(kind='bar', figsize=(12, 8))
    plt.title('Top 3 Most Used Applications')
    plt.xlabel('User')
    plt.ylabel('Total Traffic (Bytes)')
    plt.show()

def elbow_method(normalized_data, max_k=10):
    """Determine the optimized value of k using elbow method"""
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(normalized_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    return inertia
