import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

"""
Correlation Matrix Heatmap Plot

The `plot_correlation_matrix` function generates and visualizes a correlation matrix heatmap for specified columns in a DataFrame.
It calculates the correlation between the selected columns and displays it in a heatmap format, with colors representing the strength of the correlation.
Additionally, it annotates each cell with the correlation value. The function returns the correlation matrix as a DataFrame.

Args:
    data_tbl (pd.DataFrame): The input DataFrame.
    columns ([str]): A list of column names to include in the correlation analysis.

Returns:
    pd.DataFrame: The correlation matrix.
"""
def plot_correlation_matrix(data_tbl: pd.DataFrame, columns: [str]) -> pd.DataFrame:
    """
        Plot a correlation matrix heatmap for specified columns in a DataFrame.

        This function visualizes the correlation between specified columns in a DataFrame using a heatmap.

        Parameters:
        --------
        - data_tbl (pd.DataFrame): The input DataFrame.
        - columns ([str]): A list of column names to include in the correlation analoutsis.

        Returns:
        --------
        - pd.DataFrame: The correlation matrix.

        Example:
        --------
        >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        >>> data_tbl = pd.DataFrame(data)
        >>> plot_correlation_matrix(data_tbl, ['A', 'B', 'C'])
    """
    corr_matrix = data_tbl[columns].corr()
    mask = np.triu(corr_matrix)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=(34, 34))
    ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0, cmap=cmap, mask=mask, linewidths=1,
                     linecolor='white', square=True, xticklabels=True, cbar_kws={'shrink': .81})
    # Calculate p-values for each pair of variables
    degrees_of_freedom = len(corr_matrix)
    # Convert p-values matrix to a DataFrame
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if not bool(mask[i, j]):
                plt.text(j + 0.5, i + 0.5,
                         f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=16)
    # Plotting
    plt.title(f"Correlation analoutsis", fontsize=20, pad=20)
    plt.xticks(rotation=40, fontsize=17)
    plt.yticks(fontsize=17, rotation=0)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.set_outticklabels(cbar.ax.get_outticklabels(), fontsize=20)
    plt.show()
    return corr_matrix


