import pandas as pd


def main() -> pd.DataFrame:
    train_data = pd.read_csv("./data/raw/train_energy_data.csv")
    test_data = pd.read_csv("./data/raw/test_energy_data.csv")
    return train_data, test_data