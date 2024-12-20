import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def graphical_univariate_analysis(data):
    """Identify and plot the most suitable options for each variable"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(data['Dur. (ms)'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Session Duration Distribution')

    sns.histplot(data['Total DL (Bytes)'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Total Download Data Distribution')

    sns.histplot(data['Total UL (Bytes)'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Total Upload Data Distribution')

    sns.histplot(data['Total Data Volume (Bytes)'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Total Data Volume Distribution')

    plt.tight_layout()
    plt.show()