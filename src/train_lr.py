"""
Author: Rana Elladki
Date: 11/17/2025
Description: Trains and validates linear regression model.
"""
# import packages for models
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

def train_lr(X_train, y_train):
    """
    Train and fit linear regression model.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Input features for training set
    y_train : pandas.Series
        Target values for training set.
    
    Returns
    -------
    LinearRegression
        Trained and fitted linear regression model
    """
    # initiate linear regression model
    lr = LinearRegression()

    # fit the model with the training data
    lr.fit(X_train, y_train)

    return lr

def test_lr(lr, X_test, y_test):
    """
    Validate linear regression model

    Parameters
    ----------
    lr : LinearRegression
        Trained and fitted linear regression model
    X_test : pandas.DataFrame
        Input features for test set 
    y_test : pandas.Series
        True target values for training set.

    Returns
    -------
    tuple :
        (y_test_pred, report), where :
            y_test_pred : numpy.ndarray
                Prediced target values
            report : dict
                Dictionary containing statistics on predicted data:
                    - "R2"  : float
                    - "RMSE": float
                    - "MSE" : float
                    - "MAE" : float
                    - "PC"  : float 
    """
    # get predictions on test set
    y_test_pred = lr.predict(X_test)
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
