"""
Author: Rana Elladki
Date: 10/15/2025
Description: Performs exploratory data analysis on molecular descriptor
    datasets. It includes routines for correlation-based feature selection,
    outlier filtering, and log transformation of skewed data distributions. 
    After completing EDA, it splits the data into training, validation, and 
    testing sets.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def pearson_correlation(df, outdir, target='RG2', top_count=10):
    """
    Perform Pearson correlation coefficient analysis to identify descriptors
    that have the strongest relationship with the target variable (e.g., RG2).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing molecular descriptors and the target variable.
    outdir : str
        Output directory to save plots to.
    target : str, optional
        Name of the target variable column. Default is 'RG2'.
    top_count : int, optional
        Number of top correlated features to select. Default is 10.

    Returns
    -------
    pandas.DataFrame
        A reduced DataFrame containing the top correlated descriptors along
        with the target variable.

    Notes
    -----
    - The function removes the 'ID' column if present.
    - A correlation heatmap is saved to
      `<outdir>/corr_heatmap_top_<top_count>_feats.png`.
    
    Examples
    --------
    >>> reduced_df = pearson_correlation(df, target="RG2", top_count=10)
    >>> reduced_df.head()
    """
    # drop IDs because that is not a feature 
    df = df.drop(["ID"], axis=1)
    # get the correlation matrix using pearson correlation coefficient analysis
    corr_pearson = df.corr(method='pearson')
    # get the rg2 correlation column and remove rg2's correlation with itself, because it is just 1
    rg2_corr = corr_pearson[target].drop(target).abs().sort_values(ascending=False)
    # get top 20 features
    strong_corr = rg2_corr.index[:top_count]
    # reduced descriptor dataframe
    reduced_df = df[strong_corr.tolist() + [target]]
    # get a heat map
    outfile = os.path.join(outdir,f"corr_heatmap_top_{top_count}_feats.png")
    plt.figure(figsize=(10,10))
    sns.heatmap(reduced_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.savefig(outfile, dpi=300)
    return reduced_df


def filter_outliers(df, outdir, q_min=0.001, q_max=0.999):
    """ 
    This function is inspired from Day13-inclass 
    
    Filter outliers based on percentile thresholds for each feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing molecular descriptors.
    outdir : str
        Output directory to save plots to.
    q_min : float, optional
        Lower percentile threshold for filtering. Default is 0.001.
    q_max : float, optional
        Upper percentile threshold for filtering. Default is 0.999.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame with outliers removed.

    Notes
    -----
    - The function saves histograms of unfiltered and filtered data as:
      - `<outdir>/hist_unfiltered.png`
      - `<outdir>/hist_filtered.png`
    - Features dominated by zeros (e.g., NumRotatableBonds) are preserved.

    Examples
    --------
    >>> clean_df = filter_outliers(df, q_min=0.01, q_max=0.99)
    >>> clean_df.describe()
    """
    # histogram of data before filtering
    outfile = os.path.join(outdir,"hist_unfiltered.png") 
    df.hist(figsize=(10,10), bins=50)
    plt.tight_layout()  # make sure subplots don't overlap
    plt.savefig(outfile, dpi=300)

    # define a percentile value range for features
    perc_range = pd.DataFrame([df.quantile(q=q_min, axis=0), df.quantile(q=q_max, axis=0)])
    
    for feat in df.columns[:-1]:
        # these are small molecules, so most will have NumRotatableBonds=0 which will end up getting excluded, so don't remove outliers because data correctly mostly zeros
        # include features that have values greater than or equal to bottom 0.1% of data
        df = df[df[feat] >= perc_range[feat].iloc[0,]]
        # include features that have values less than or equal to top 99% of data
        df = df[df[feat] <= perc_range[feat].iloc[1,]]
    
    outfile = os.path.join(outdir,"hist_filtered.png")
    df.hist(figsize=(10,10), bins=50)
    plt.tight_layout()  # make sure subplots don't overlap
    plt.savefig(outfile, dpi=300)
    
    return df

def log_transform(df, outdir, exclude=None, skew_thresh=1.0):
    """
    Apply log2 transformation to columns with high positive skewness.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing molecular descriptors.
    outdir : str
        Output directory to save plots to.
    exclude : list of str, optional
        Columns to exclude from transformation. Default is None.
    skew_thresh : float, optional
        Threshold above which skewed columns are log-transformed.
        Default is 1.0.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with skewed columns log-transformed.

    Notes
    -----
    - The function only applies the transformation to columns where
      all values are positive.
    - A histogram of the transformed data is saved as:
      `<outdir>/hist_logged.png`.

    Examples
    --------
    >>> logged_df = log_transform(df, exclude=["RG2"])
    >>> logged_df.head()
    """
    # calculate skewness for each column
    skewness = df[df.columns].skew()
    for col in df.columns:
        if exclude!=None and col in exclude:
            continue
        if skewness[col] >= skew_thresh and (df[col] > 0).all():
            df[col] = np.log2(df[col])
    
    outfile = os.path.join(outdir,"hist_logged.png")
    df.hist(figsize=(10,10), bins=50)
    plt.tight_layout()  # make sure subplots don't overlap
    plt.savefig(outfile, dpi=300)
    
    return df

def split_data(df, rand_state=42):
    """
    Split the dataset into training, validation, and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset including target column 'RG2'.
    rand_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)

    Examples
    --------
    >>> splits = split_data(desc_df, rand_state=123)
    >>> len(splits)
    6
    """
    # drop RG2 because that is what we want to predict
    X = df.drop(["RG2"], axis=1)
    y = df["RG2"]

    # split into training/validation and test set (80:20)
    X_train_val, X_test, y_train_val, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=rand_state)
    
    # split training/validation set into separate training and validation sets (75:25)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=rand_state)
    
    data_split = (X_train, X_val, X_test, y_train, y_val, y_test)

    return data_split