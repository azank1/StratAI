import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

#######################################
# Utility Functions
#######################################
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

def get_smieo_signal(df):
    """
    Loads settings from "settings/smi_ergodic_settings.json" and computes a binary signal using only xSMI.
    Returns 1 if xSMI > thresh; 0 if xSMI < -thresh; otherwise, hold the previous signal.
    """
    settings = load_json_settings("settings/smi_ergodic_settings.json")
    fastPeriod = int(settings.get("fastPeriod", 4))
    slowPeriod = int(settings.get("slowPeriod", 8))
    SmthLen = int(settings.get("SmthLen", 3))
    thresh = float(settings.get("thresh", 0.1))
    source = settings.get("source", "close")
    
    def compute_smi_ergodic_indicator(df, fastPeriod, slowPeriod, SmthLen, source):
        xPrice = df[source].copy()
        xPrice1 = xPrice - xPrice.shift(1)
        xPrice2 = xPrice.diff().abs()
        xSMA_R = xPrice1.ewm(span=fastPeriod, adjust=False).mean().ewm(span=slowPeriod, adjust=False).mean()
        xSMA_aR = xPrice2.ewm(span=fastPeriod, adjust=False).mean().ewm(span=slowPeriod, adjust=False).mean()
        xSMI = (xSMA_R / xSMA_aR).fillna(0)
        xEMA_SMI = xSMI.ewm(span=SmthLen, adjust=False).mean()  # for reference only
        return xSMI, xEMA_SMI

    xSMI, _ = compute_smi_ergodic_indicator(df, fastPeriod, slowPeriod, SmthLen, source)
    n = len(df)
    final_signal = np.zeros(n, dtype=int)
    final_signal[0] = 0
    for i in range(1, n):
        if xSMI.iloc[i] > thresh:
            final_signal[i] = 1
        elif xSMI.iloc[i] < -thresh:
            final_signal[i] = 0
        else:
            final_signal[i] = final_signal[i-1]
    return np.array(final_signal, dtype=int)
