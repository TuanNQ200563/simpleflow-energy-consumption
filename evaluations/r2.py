from sklearn.metrics import r2_score


def scorer(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    float: The RMSE value.
    """
    return r2_score(y_true, y_pred)