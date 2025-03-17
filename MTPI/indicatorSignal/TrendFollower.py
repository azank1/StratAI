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


def get_trend_follower_signal(df):
    """
    Loads settings from "settings/trend_follower_trained_settings.json" and computes a binary signal.
    Computes the TrendFollower indicator as defined in your Pinescript logic, then returns 1 if positive, 0 if negative.
    """
    settings = load_json_settings("settings/trend_follower_trained_settings.json")
    prd = int(settings.get("prd", 20))
    maprd = int(settings.get("maprd", 20))
    rateinp = float(settings.get("rateinp", 1))
    linprd = int(settings.get("linprd", 5))
    matype = settings.get("matype", "EMA")
    ulinreg = settings.get("ulinreg", True)
    
    def wma(series, window):
        weights = np.arange(1, window+1)
        return series.rolling(window, min_periods=1).apply(lambda x: np.dot(x, weights[:len(x)])/weights[:len(x)].sum(), raw=True)
    
    def rolling_linreg(series, window):
        def linreg(x):
            if len(x) < 2 or np.isnan(x).any():
                return np.nan
            idx = np.arange(len(x))
            slope, intercept = np.polyfit(idx, x, 1)
            return slope * (len(x) - 1) + intercept
        return series.rolling(window=window, min_periods=2).apply(linreg, raw=True)
    
    def get_moving_average(series, matype, period, df=None):
        if matype == 'EMA':
            return series.ewm(span=period, adjust=False).mean()
        elif matype == 'SMA':
            return series.rolling(window=period, min_periods=1).mean()
        elif matype == 'RMA':
            return series.ewm(span=period, adjust=False).mean()
        elif matype == 'WMA':
            return wma(series, period)
        elif matype == 'VWMA':
            if df is not None and 'volume' in df.columns:
                return (series * df['volume']).rolling(window=period, min_periods=1).sum() / \
                       df['volume'].rolling(window=period, min_periods=1).sum()
            else:
                return series.rolling(window=period, min_periods=1).mean()
        else:
            return series.rolling(window=period, min_periods=1).mean()
    
    def compute_trend_follower(df, matype, prd, maprd, rateinp, ulinreg, linprd):
        rate = rateinp / 100.0
        high_280 = df['close'].rolling(window=280, min_periods=1).max()
        low_280  = df['close'].rolling(window=280, min_periods=1).min()
        pricerange = high_280 - low_280
        chan = pricerange * rate
        masrc = get_moving_average(df['close'], matype, maprd, df)
        if ulinreg:
            masrc_lin = rolling_linreg(masrc, linprd)
            ma = masrc_lin
        else:
            ma = masrc
        hh = ma.rolling(window=prd, min_periods=1).max()
        ll = ma.rolling(window=prd, min_periods=1).min()
        diff = (hh - ll).abs()
        condition = diff > chan
        trend = np.where(
            condition,
            np.where(ma > (ll + chan), 1, np.where(ma < (hh - chan), -1, 0)),
            0
        )
        trend = pd.Series(trend, index=df.index)
        safe_chan = chan.replace(0, np.nan)
        indicator = trend * diff / safe_chan
        indicator = indicator.fillna(0)
        return indicator
    
    indicator = compute_trend_follower(df, matype, prd, maprd, rateinp, ulinreg, linprd)
    binary_signal = (indicator > 0).astype(int)
    return np.array(binary_signal, dtype=int)