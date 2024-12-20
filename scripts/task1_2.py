import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def load_data(engine):
    """Load the cleaned data from the database"""
    query = """
    SELECT "MSISDN/Number" AS user,
           "Bearer Id",
           "Dur. (ms)",
           "Total DL (Bytes)",
           "Total UL (Bytes)"
    FROM xdr_data;
    """
    data = pd.read_sql_query(query, engine)
    return data

def treat_missing_values_and_outliers(data):
    """Treat missing values and outliers by replacing them with the mean"""
    data.fillna(data.mean(), inplace=True)
    for col in data.columns:
        if data[col].dtype != 'object':
            mean = data[col].mean()
            std = data[col].std()
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std
            data[col] = np.where(data[col] > upper_bound, mean, data[col])
            data[col] = np.where(data[col] < lower_bound, mean, data[col])
    return data

def describe_variables(data):
    """Describe all relevant variables and associated data types"""
    descriptions = data.describe(include='all')
    return descriptions

def analyze_basic_metrics(data):
    """Analyze basic metrics such as mean, median, etc."""
    basic_metrics = data.describe()
    return basic_metrics

def non_graphical_univariate_analysis(data):
    """Compute dispersion parameters for each quantitative variable"""
    desc_stats = data.describe().loc[['std', 'min', 'max', '25%', '50%', '75%']]
    var_stats = pd.DataFrame(data.var(), columns=['var']).T
    dispersion_params = pd.concat([desc_stats, var_stats])
    return dispersion_params

def bivariate_analysis(data):
    """Explore the relationship between each application & the total DL+UL data"""
    sns.pairplot(data[['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Total Data Volume (Bytes)']])
    plt.show()

def correlation_analysis(data):
    """Compute a correlation matrix for the specified variables"""
    correlation_matrix = data[['Total DL (Bytes)', 'Total UL (Bytes)', 'Social Media DL (Bytes)', 'Google DL (Bytes)',
                               'Email DL (Bytes)', 'YouTube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    return correlation_matrix

def dimensionality_reduction(data):
    """Perform PCA for dimensionality reduction and interpret the results"""
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data[['Total DL (Bytes)', 'Total UL (Bytes)', 'Social Media DL (Bytes)', 'Google DL (Bytes)',
                                       'Email DL (Bytes)', 'YouTube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL']])
    pca_df = pd.DataFrame(data=pca_data, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['user'] = data['user']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df)
    plt.title('PCA Result')
    plt.show()

    return pca_df