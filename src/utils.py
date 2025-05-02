import pandas as pd

def load_data(filepath):
    """
    Load a CSV file and return a DataFrame.
    """
    return pd.read_csv(filepath)
