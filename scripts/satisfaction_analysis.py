import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def replace_missing_values_and_outliers(cleaned_data):
    """
    Clean data by replacing missing values and handling outliers.
    """
    columns = [
        "TCP DL Retrans. Vol (Bytes)",
        "TCP UL Retrans. Vol (Bytes)",
        "Avg RTT DL (ms)",
        "Avg RTT UL (ms)",
        "Avg Bearer TP DL (kbps)",
        "Avg Bearer TP UL (kbps)"
    ]

    for column in columns:
        mean_value = cleaned_data[column].mean()
        std_value = cleaned_data[column].std()
        cleaned_data[column] = cleaned_data[column].fillna(mean_value)
        cleaned_data[column] = np.where(
            cleaned_data[column] > mean_value + 3 * std_value,
            mean_value, cleaned_data[column]
        )

    return cleaned_data

def aggregate_engagement_metrics(cleaned_data):
    """
    Aggregate engagement metrics by IMSI.

    Parameters:
    cleaned_data (pd.DataFrame): DataFrame containing the cleaned data.

    Returns:
    pd.DataFrame: A DataFrame with aggregated engagement metrics.
    """
    aggregated_data = cleaned_data.groupby('IMSI').agg({
        'Bearer Id': 'count',
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).rename(columns={
        'Bearer Id': 'session_frequency',
        'Dur. (ms)': 'total_duration',
        'Total DL (Bytes)': 'total_download',
        'Total UL (Bytes)': 'total_upload'
    }).reset_index()

    return aggregated_data

def aggregate_experience_metrics(cleaned_data):
    """
    Aggregate experience metrics by IMSI.

    Parameters:
    cleaned_data (pd.DataFrame): DataFrame containing the cleaned data.

    Returns:
    pd.DataFrame: A DataFrame with aggregated experience metrics.
    """
    cleaned_data['avg_tcp_retransmission'] = cleaned_data["TCP DL Retrans. Vol (Bytes)"] + cleaned_data["TCP UL Retrans. Vol (Bytes)"]
    cleaned_data['avg_rtt'] = cleaned_data["Avg RTT DL (ms)"] + cleaned_data["Avg RTT UL (ms)"]
    cleaned_data['avg_throughput'] = cleaned_data["Avg Bearer TP DL (kbps)"] + cleaned_data["Avg Bearer TP UL (kbps)"]

    aggregated_data = cleaned_data.groupby('IMSI').agg({
        'avg_tcp_retransmission': 'mean',
        'avg_rtt': 'mean',
        'avg_throughput': 'mean'
    }).reset_index()

    return aggregated_data


def normalize_and_cluster(aggregated_data, n_clusters=3):
    """
    Normalize data and perform K-means clustering.

    Parameters:
    aggregated_data (pd.DataFrame): DataFrame containing the aggregated data.
    n_clusters (int): Number of clusters for K-means.

    Returns:
    tuple: Updated DataFrame, scaler, and KMeans model.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(aggregated_data[['session_frequency', 'total_duration', 'total_download', 'total_upload']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(normalized_data)
    aggregated_data['cluster'] = clusters

    return aggregated_data, scaler, kmeans

def calculate_distance(points, centroid):
    """
    Calculate Euclidean distances from points to a centroid.

    Parameters:
    points (pd.DataFrame): DataFrame of points.
    centroid (np.array): Centroid to calculate distances from.

    Returns:
    np.array: Array of distances.
    """
    return euclidean_distances(points, centroid.reshape(1, -1))

def assign_scores(data, engagement_centroid, experience_centroid):
    """
    Assign engagement and experience scores to data based on centroids.

    Parameters:
    data (pd.DataFrame): DataFrame to assign scores to.
    engagement_centroid (np.array): Engagement centroid.
    experience_centroid (np.array): Experience centroid.

    Returns:
    pd.DataFrame: DataFrame with assigned scores.
    """
    engagement_scores = calculate_distance(data[['session_frequency', 'total_duration', 'total_download', 'total_upload']], engagement_centroid)
    experience_scores = calculate_distance(data[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']], experience_centroid)

    data['engagement_score'] = engagement_scores
    data['experience_score'] = experience_scores

    return data

def compute_satisfaction_score(data):
    """
    Compute satisfaction score as the average of engagement and experience scores.

    Parameters:
    data (pd.DataFrame): DataFrame containing engagement and experience scores.

    Returns:
    pd.DataFrame: DataFrame with an additional satisfaction_score column.
    """
    data['satisfaction_score'] = data[['engagement_score', 'experience_score']].mean(axis=1)
    return data


def build_regression_model(data):
    """
    Build a linear regression model to predict satisfaction score.

    Parameters:
    data (pd.DataFrame): DataFrame containing engagement and experience scores, and satisfaction score.

    Returns:
    tuple: The trained model and the mean squared error of the predictions.
    """
    X = data[['engagement_score', 'experience_score']]
    y = data['satisfaction_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse

def run_kmeans_clustering(data, n_clusters=2):
    """
    Run K-means clustering on engagement and experience scores.

    Parameters:
    data (pd.DataFrame): DataFrame containing engagement and experience scores.
    n_clusters (int): Number of clusters for K-means.

    Returns:
    pd.DataFrame: DataFrame with an additional 'cluster' column.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['cluster'] = kmeans.fit_predict(data[['engagement_score', 'experience_score']])
    return data

def aggregate_scores_per_cluster(data):
    """
    Aggregate satisfaction and experience scores per cluster.

    Parameters:
    data (pd.DataFrame): DataFrame containing cluster assignments and scores.

    Returns:
    pd.DataFrame: DataFrame with average scores per cluster.
    """
    cluster_agg = data.groupby('cluster').agg({
        'satisfaction_score': 'mean',
        'experience_score': 'mean'
    }).reset_index()
    return cluster_agg