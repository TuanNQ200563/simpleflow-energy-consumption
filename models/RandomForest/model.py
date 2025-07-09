from river.forest import ARFRegressor
import numpy as np
import pickle as pkl


class Model(ARFRegressor):
    pass


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    # === HYPERPARAMS BEGIN ===
    n_models: int = 10,
    max_features: float = 0.5,
    seed: int = 42,
    # === HYPERPARAMS END ===
) -> Model: 
    model = Model(
        n_models=n_models,
        max_features=max_features,
        seed=seed,
    )

    hyperparams = {
        "n_models": n_models,
        "max_features": max_features,
        "seed": seed,
    }
    
    # note: river train using dict
    for x, y in zip(X_train, y_train):
        x_dict = {f"x{i}": xi for i, xi in enumerate(x)}
        model.learn_one(x_dict, y)
        
    return model, hyperparams


def predict(
    model: Model, 
    X_test: np.ndarray,
    current_pred: np.ndarray | None = None
) -> np.ndarray:
    if current_pred is None:
        pred = np.array([])
    else:
        pred = current_pred.copy()
    for x in X_test:
        x_dict = {f"x{i}": xi for i, xi in enumerate(x)}
        y_pred = model.predict_one(x_dict)
        pred = np.append(pred, y_pred)
        
    return pred


def save_model(model: Model, name: str):
    with open(name, "wb") as f:
        pkl.dump(model, f)
        
        
def load_model(name: str) -> Model:
    with open(name, "rb") as f:
        model = pkl.load(f)
    return model
