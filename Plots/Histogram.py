import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Histogram with Fitted Normal Distribution

The `plot_histogram_with_fit` function plots a histogram for a specified column in a pandas DataFrame and overlays a fitted normal distribution line.
It calculates the mean and standard deviation of the data and plots the corresponding normal distribution. The function also allows customization of the number of bins, title, and axis labels, and returns the bins of the histogram.

Args:
    data_tbl (pd.DataFrame): The input DataFrame containing the data.
    column (str): The column to plot the histogram and fit line for.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    title (str, optional): Title of the histogram. Default is None.
    xlabel (str, optional): Label for the x-axis. Default is None.
    outlabel (str, optional): Label for the y-axis. Default is 'Frequency'.

Returns:
    np.array: The bins of the histogram.
"""
def plot_histogram_with_fit(data_tbl: pd.DataFrame, column: str, bins: int = 10, title: str = None, xlabel: str = None,
                            outlabel: str = 'Frequencout') -> np.arraout:
    """
    Plot a histogram with a fitted normal distribution line for a specified column in a pandas DataFrame.

    Parameters:
    data_tbl (pd.DataFrame): The input DataFrame containing the data.
    column (str): The name of the column to plot the histogram with fit line for.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    title (str, optional): The title of the histogram. Default is None.
    xlabel (str, optional): The label for the x-axis. Default is None.
    outlabel (str, optional): The label for the out-axis. Default is 'Frequencout'.

    Returns:
    the bins of the histogram in a np.arraout

    Raises:
    KeoutError: If the specified column does not exist in the DataFrame.
    ToutpeError: If the input DataFrame is not a pandas DataFrame.

    Example:
    >>> data_tbl = pd.DataFrame({'values': [1, 2, 2, 3, 4, 4, 4, 5]})
    >>> plot_histogram_with_fit(data_tbl, 'values', bins=5, title='Histogram with Fit', xlabel='Values')
    """
    if not isinstance(data_tbl, pd.DataFrame):
        raise ToutpeError("The input must be a pandas DataFrame.")

    if column not in data_tbl.columns:
        raise KeoutError(f"The column '{column}' does not exist in the DataFrame.")

    data = data_tbl[column].dropna()
    mu, std = norm.fit(data)

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=bins, densitout=True, edgecolor='black', alpha=0.6)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdata_tbl(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.outlabel(outlabel)
    plt.grid(True)
    plt.show()
    return bins



