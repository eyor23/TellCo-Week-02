import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def bivariate_analysis(data):
    """Explore the relationship between each application & the total DL+UL data"""
    sns.pairplot(data[['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Total Data Volume (Bytes)']])
    plt.show()
