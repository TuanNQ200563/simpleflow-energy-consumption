from sklearn.metrics import root_mean_squared_error


def scorer(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    float: The RMSE value.
    """
    return root_mean_squared_error(y_true, y_pred)