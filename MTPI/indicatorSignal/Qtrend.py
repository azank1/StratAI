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

def get_qtrend_signal(df):
    """
    Loads settings from "settings/qtrend_settings.json" and computes a binary signal.
    Uses a channel approach with ATR. Returns 1 if close > channel+epsilon, 0 if close < channel-epsilon.
    Otherwise, it holds the previous value.
    """
    settings = load_json_settings("settings/qtrend_settings.json")
    source = settings.get("source", "close")
    p = int(settings.get("p", 200))
    atr_p = int(settings.get("atr_p", 14))
    mult = float(settings.get("mult", 1.0))
    
    close = df[source]
    high = df['high']
    low = df['low']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_p, min_periods=1).mean().shift(1)
    epsilon = mult * atr
    m = (close.rolling(window=p, min_periods=1).max() + close.rolling(window=p, min_periods=1).min()) / 2
    
    signal = np.where(close > m + epsilon, 1, np.where(close < m - epsilon, 0, np.nan))
    signal = pd.Series(signal, index=df.index).ffill().fillna(0).astype(int)
    return np.array(signal, dtype=int)