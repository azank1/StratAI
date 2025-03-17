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

df = pd.read_csv("./CSVdata/target.csv")
df['DateTime'] = pd.to_datetime(df['time'], unit='s')
df.sort_values('DateTime', inplace=True)
df.set_index('DateTime', inplace=True)
df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
target = df['manual_signal'].values
# For aggregation we also create a date column (daily)
df['date'] = df.index.date


def backtest_equity(df, binary_signal):
    """
    Simulates an equity curve based on the binary signal.
    When signal == 1, assume full investment (compounded return);
    when 0, equity remains flat.
    """
    close_prices = df['close'].astype(float).values
    equity = [1.0]
    for i in range(1, len(close_prices)):
        if binary_signal[i] == 1:
            equity.append(equity[-1] * (close_prices[i] / close_prices[i-1]))
        else:
            equity.append(equity[-1])
    return np.array(equity)

def plot_backtest(df, binary_signal, equity, title_prefix="Aggregated Indicator"):
    """
    Plots the BTC price with vertical markers for aggregated signal transitions,
    and the simulated equity curve.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10), sharex=True)
    
    # Plot price chart with signal markers.
    ax1.plot(df.index, df['close'], label='BTC Price', color='black', linewidth=1.5)
    transitions = np.where(np.diff(binary_signal) != 0)[0] + 1
    for idx in transitions:
        dt = df.index[idx]
        if binary_signal[idx] == 1:
            ax1.axvline(x=dt, color='green', linestyle='--', alpha=0.7)
        else:
            ax1.axvline(x=dt, color='red', linestyle='--', alpha=0.7)
    ax1.set_title(f"{title_prefix} Signals on BTC Price Chart")
    ax1.set_ylabel("Price")
    ax1.legend()
    
    # Plot equity curve.
    ax2.plot(df.index, equity, label='Equity Curve', color='orange', linewidth=2)
    ax2.set_title(f"{title_prefix} Equity Curve")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Equity")
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#######################################
# Indicator Signal Functions
#######################################

# 1. HullSuite Indicator Signal
def get_hull_suite_signal(df):
    """
    Loads settings from "settings/hull_suite_settings.json" and computes a binary signal.
    The method computes a Hull Moving Average (HMA) using a weighted moving average.
    Returns 1 if the current HMA is greater than the HMA shifted by 2 bars, else 0.
    """
    settings = load_json_settings("./MTPI/settings/hull_suite_settings.json")
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

# 2. QTrend Indicator Signal
def get_qtrend_signal(df):
    """
    Loads settings from "settings/qtrend_settings.json" and computes a binary signal.
    Uses a channel approach with ATR. Returns 1 if close > channel+epsilon, 0 if close < channel-epsilon.
    Otherwise, it holds the previous value.
    """
    settings = load_json_settings("./MTPI/settings/qtrend_settings.json")
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

# 3. SMIEO Indicator Signal
def get_smieo_signal(df):
    """
    Loads settings from "settings/smi_ergodic_settings.json" and computes a binary signal using only xSMI.
    Returns 1 if xSMI > thresh; 0 if xSMI < -thresh; otherwise, hold the previous signal.
    """
    settings = load_json_settings("./MTPI/settings/smi_ergodic_settings.json")
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

# 4. TrendFollower Indicator Signal
def get_trend_follower_signal(df):
    """
    Loads settings from "settings/trend_follower_trained_settings.json" and computes a binary signal.
    Computes the TrendFollower indicator as defined in your Pinescript logic, then returns 1 if positive, 0 if negative.
    """
    settings = load_json_settings("./MTPI/settings/trend_follower_settings.json")
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

#######################################
# 5. Aggregating Signals & Backtesting the Aggregated Strategy
#######################################
def aggregate_signals(signal_list):
    """
    Given a list of binary signal arrays (from each indicator), aggregate them by taking the average.
    Then, threshold at 0.5: if the average > 0.5, aggregated signal = 1; else 0.
    """
    # Stack signals to form a 2D array: shape (num_indicators, n)
    signals = np.vstack(signal_list)
    aggregated = np.mean(signals, axis=0)
    final_aggregated = (aggregated > 0.5).astype(int)
    return final_aggregated

def backtest_aggregated(df, aggregated_signal):
    """
    Simulates the equity curve using the aggregated binary signal.
    """
    return backtest_equity(df, aggregated_signal)




#######################################
# 6. Aggregate Signals from All 4 Indicators and Plot Backtest
#######################################
# Compute binary signals for each indicator
hull_signal = get_hull_suite_signal(df)
qtrend_signal = get_qtrend_signal(df)
smieo_signal = get_smieo_signal(df)
trend_follower_signal = get_trend_follower_signal(df)

# Aggregate the signals
aggregated_signal = aggregate_signals([hull_signal, qtrend_signal, smieo_signal, trend_follower_signal])

# Simulate equity curve using the aggregated signal
equity_curve = backtest_aggregated(df, aggregated_signal)

# Plot the results (price chart with aggregated signal transitions and equity curve)
plot_backtest(df, aggregated_signal, equity_curve, title_prefix="Aggregated Indicator")

print(equity_curve[-1]) 