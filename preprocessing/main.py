import pandas as pd
from typing import Union

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


def build_encoder():
    features = ["Building Type", "Day of Week"]
    encoder = ColumnTransformer(
        transformers=[(
            "cat", 
            OrdinalEncoder(
                categories=[
                    ["Residential", "Commercial", "Industrial"],
                    ["Weekday", "Weekend"]
                ],
            ), 
            features
        )],
        remainder="passthrough",
    )
    return encoder


def transform_data(
    data: Union[
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame]
    ]
):
    train_data, test_data = data
    encoder = build_encoder()
    
    X_train = train_data.drop(columns=["Energy Consumption"])
    y_train = train_data["Energy Consumption"]
    
    X_test = test_data.drop(columns=["Energy Consumption"])
    y_test = test_data["Energy Consumption"]
    
    encoder.fit(X_train)
    X_train_encoded = encoder.transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    
    transformer = {
        "features": encoder,
        "target": None,
    }
    
    return X_train_encoded, X_test_encoded, y_train, y_test, transformer