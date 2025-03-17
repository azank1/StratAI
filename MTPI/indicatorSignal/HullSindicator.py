import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

def load_json_settings(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def load_target_data(csv_path="./CSVdata/target.csv"):
    """
    Loads the target CSV (daily data) which must include columns:
    time, close, and manual_signal.
    """
    df = pd.read_csv(csv_path)
    df['DateTime'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
    return df
    
def get_hull_suite_signal(df):
    """
    Loads settings from "settings/hull_suite_settings.json" and computes a binary signal.
    The method computes a Hull Moving Average (HMA) using a weighted moving average.
    Returns 1 if the current HMA is greater than the HMA shifted by 2 bars, else 0.
    """
    settings = load_json_settings("settings/hull_suite_settings.json")
    source = settings.get("source", "close")
    length = int(settings.get("length", 55))
    
    def wma(series, window):
        weights = np.arange(1, window+1)
        return series.rolling(window, min_periods=1).apply(
            lambda x: np.dot(x, weights[:len(x)])/weights[:len(x)].sum(), raw=True)
    
    half_length = int(round(length/2))
    sqrt_length = int(round(np.sqrt(length)))
    hma = wma(2 * wma(df[source], half_length) - wma(df[source], length), sqrt_length)
    binary_signal = (hma > hma.shift(2)).astype(int)
    return np.array(binary_signal, dtype=int)
