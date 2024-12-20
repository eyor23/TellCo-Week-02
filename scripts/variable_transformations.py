import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def variable_transformations(data):
    """Segment users into decile classes and compute total data per decile class"""
    data['Total Data Volume (Bytes)'] = data['Total DL (Bytes)'] + data['Total UL (Bytes)']
    data['Decile'] = pd.qcut(data['Dur. (ms)'], 10, labels=False, duplicates='drop')
    decile_data = data.groupby('Decile').agg({ 'Total Data Volume (Bytes)': 'sum' }).reset_index()
    return data, decile_data