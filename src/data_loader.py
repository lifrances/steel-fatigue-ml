import os
import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(data):
    """
    Splits data into features (X) and target (y), then performs an 80/20 train-test split.
    """
    X = data.drop(columns=['Fatigue'])
    y = data['Fatigue']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_all_datasets(data_path):
    """
    Loads all experimental CSV files and returns a dictionary of data splits.
    """
    files = {
        'Original': 'original_data.csv',
        'Preprocessed': 'preprocessed_data.csv',
        'no_HJP': 'preprocessed_data_no_HJP.csv',
        'no_CE': 'preprocessed_data_no_CE.csv',
        'no_sixthC': 'preprocessed_data_no_sixthC.csv',
        'no_logC': 'preprocessed_data_no_logC.csv'
    }

    datasets = {}
    for name, filename in files.items():
        file_path = os.path.join(data_path, filename)
        df = pd.read_csv(file_path)
        datasets[name] = data_split(df)
    return datasets
