import pandas as pd

def non_graphical_univariate_analysis(data):
    """Compute dispersion parameters for each quantitative variable"""
    desc_stats = data.describe().loc[['std', 'min', 'max', '25%', '50%', '75%']]
    var_stats = pd.DataFrame(data.var(), columns=['var']).T
    dispersion_params = pd.concat([desc_stats, var_stats])
    return dispersion_params
