"""
Author: Rana Elladki
Date: 11/17/2025
Description: Trains and validates artificial neural network.
"""
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr
import pandas as pd

def train_val_ann(X_train, X_val, y_train, y_val, rand_state=42):
    """
    Train and valid an artificial neural network using GridSearchCV.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Input features for training set
    X_val : pandas.DataFrame
        Input features for validation set
    y_train : pandas.Series
        Target values for training set.
    y_val : pandas.Series
        Target values for valdiation set.
    rand_state : int, optional
        Random seed for MLPRegressor. Default is 42.

    Returns
    -------
    GridSearchCV
        GridSearchCV object containing the best fitted MLPRegressor model.
    """
    # combine train and validation set for cross validation
    X_comb = pd.concat([X_train, X_val], ignore_index=True)
    y_comb = pd.concat([y_train, y_val], ignore_index=True)

    # hyperparameter options for search
    param_grid = {'hidden_layer_sizes': [(10,), (10, 20), (20, 30), (10, 20, 30)],
                   'batch_size': [100, 200, 500],
                   'learning_rate_init': [0.01, 0.05, 0.1]}
    
    # define baseline MLP model
    ann = MLPRegressor(activation='relu', solver='adam', random_state=rand_state)

    # run grid search with 5-fold cross validation
    ann_gs = GridSearchCV(ann, param_grid, cv=5, verbose=1, scoring='r2', n_jobs=-1)
    ann_gs.fit(X_comb, y_comb)
    return ann_gs

def test_ann(ann_gs, X_test, y_test):
    """
    Validate trained artificial neural network model

    Parameters
    ----------
    ann_gs : GridSearchV
        Fitted GridSearchCV object containing the best ANN model.
    X_test : pandas.DataFrame
        Input features for test set 
    y_test : pandas.Series
        True target values for testing set.

    Returns
    -------
    tuple :
        (y_test_pred, report), where :
            y_test_pred : numpt.ndarray 
                Prediced target values
            report : dict
                Dictionary containing statistics on predicted data
                    - "R2"  : float
                    - "RMSE": float
                    - "MSE" : float
                    - "MAE" : float
                    - "PC"  : float 
    """
    # get predictions on test set
    y_test_pred = ann_gs.predict(X_test)
    # get pearson's rank correlation between true and predicted values
    pearson_corr = pearsonr(y_test, y_test_pred)[0]
    r2 = r2_score(y_test, y_test_pred) # get R^2 (coefficient of determination)
    rmse = root_mean_squared_error(y_test, y_test_pred) # get root mean squared error
    mse = mean_squared_error(y_test, y_test_pred) # get mean sequared error
    mae = mean_absolute_error(y_test, y_test_pred) # get mean absoltute error
    # save statistics to dictionary 
    report = {
        "R2": r2,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "PC": pearson_corr
    }
    return y_test_pred, report

