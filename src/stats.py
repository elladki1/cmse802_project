"""
Author: Rana Elladki
Date: 11/17/2025
Description: Utilities for creating plots and PrettyTable summaries of model 
    performance statistics, including true vs predicted plots, linear regression 
    coefficient visualization, SHAP analysis, and table generation for model metrics.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable

def plot_true_pred(ax, y_true, y_pred, plt_label=None):
    """
    Plot true versus predicted values use a hexbin density map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    y_true : array-like
        Array of true target values.
    y_pred : array-like
        Array of predicted target values
    plt_lable : str, optional
        Optional lable to place on the top-left of the plot.
    
    Returns
    -------
    None
    """
    # Determine plot limits based on the min/max of true/predicted values
    x_range = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    y_range = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    
    # Use hexbin to visualize point density instead of scatter
    hb = ax.hexbin(y_true, y_pred, gridsize=30, cmap='viridis', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label("Number of Points")
    ax.set_xlabel('True Val')
    ax.set_ylabel('Predicted Val')
    ax.plot(x_range, y_range, 'r--')
    ax.grid()
    if plt_label is not None:
        ax.text(
            0.05, 0.95, plt_label, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

def lr_coef_weights(model, feats, outfile):
    """
    Plot and save linear regression feature coefficients bar plot.

    Parameters
    ----------
    model : sklearn.linear_model
        Trained linear regression model.
    feats : list of str
        List of feature names corresponding to model coefficients.
    outfile : str
        File path to save the coefficients bar plot to.
    
    Returns
    -------
    None
    """
    coef_df = pd.DataFrame({'Feature': feats, 'Coefficient': model.coef_})
    
    # Sort by absolute coefficient value in decending order
    #   Sorting by absolute magnitude to highlight strongest contributors
    coef_df['Abs_Coef'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coef', ascending=False).reset_index(drop=True)
    
    # Create a visual to compare feature coefficients
    plt.bar(coef_df['Feature'], coef_df['Coefficient'])
    # Rotated x-axis labels for readability
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Value')
    plt.title('Linear Regression Feature Coefficients')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def ann_shap_analysis(model, X_train, X_test, outfile, sample=500):
    """
    Run SHAP analysis on neural network model and save a summary plot.

    Parameters
    ----------
    model : sklearn.neural_network
        Trained artificial neural network model.
    X_train : pandas.DataFrame
        Training data used to initialize SHAP explainer.
    X_test : pandas.DataFrame
        Testing data to use a subset of for SHAP analysis.
    outfile : str
        File path to save the SHAP summary plot to.
    sample: int
        Sample size for SHAP calculations.

    Returns
    -------
    None
    """
    # Create a SHAP explainer for the model
    explainer = shap.Explainer(model.predict, X_train)

    # Calculate SHAP values for the test set (sample 500 points)
    #   500 sample points randomly chosen to reduce computation time
    sample_indices = np.random.choice(X_test.shape[0], sample, replace=False)
    X_test_sample = X_test.iloc[sample_indices]
    shap_vals = explainer.shap_values(X_test_sample)
    
    # Generate SHAP summary plot showing feature importance and impact
    plt.figure()
    shap.summary_plot(shap_vals, X_test_sample, feature_names=X_test.columns)
    plt.savefig(outfile)
    plt.close()

def populate_table(dict, table):
    """
    Populate a Prettytable with model performance metrics

    Parameters
    ----------
    dict : dict
        Dictionary of dictionaries where the keys encode model type and scaling,
        and the values are dictionaries of statistical metrics R2, RMSE, MSE, MAE, 
        and PC.
    table : PrettyTable
        PrettyTable object to add rows to.

    Returns
    -------
    PrettyTable 
        The updated Prettytable with populated rows. 
    """
    stats_metrics = ["R2", "RMSE", "MSE", "MAE", "PC"]
    # Loop through the dictionary and populate the table
    for key, metrics in dict.items():
        # Extract model type and scaling method from dictionary key string
        parts = key.split(maxsplit=1)
        # Print LR as Linear Regression and ANN as artificial neural network for readability
        if parts[0]=="LR":
            model = "Linear Regression (LR)"
        elif parts[0]=="ANN":
            model = "Artificial Neural Network (ANN)"
        else:
            model = 'NaN'
        
        scaling = parts[1] if len(parts) > 1 else "--"

        row_vals = []

        # Add row with formatted values (e.g., 3 decimal places)
        for stat in stats_metrics:
            if stat in metrics:
                row_vals.append(f"{metrics[stat]:.3f}")
            else:
                # If metric is missing, print NaN in row
                row_vals.append("--")

        table.add_row([model, scaling] + row_vals)

    return table 

def create_table(valid_dict, test_dict):
    """
    Create a PrettyTable summarizing validation and test model metrics

    Parameters
    ----------
    valid_dict : dict
        Dictionary of validation metrics for different models and scalings.
    test_dict : dict
        Dictionary of test metrics for different models and scalings
    
    Returns
    -------
    PrettyTable
        A formatted PrettyTable containing statatistics for validation and test sets.
    """
    # Create table with headers
    table = PrettyTable(["Model", "Scaling", "R²", "RMSE", "MSE", "MAE", "PC"])

    table.add_row(["--- VALIDATION SET ---", "", "", "", "", "", ""])
    # Loop through the dictionary and populate the table with validation statistics
    populate_table(valid_dict, table)
    
    table.add_row(["--- TESTING SET ---", "", "", "", "", "", ""])
    # Loop through the dictionary and populate the table with testing statistics
    populate_table(test_dict, table)

    return table
