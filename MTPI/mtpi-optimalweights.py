import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os
from bayes_opt import BayesianOptimization

#######################################
# Utility Functions
#######################################
def load_json_settings(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def load_target_data(csv_path="./CSVdata/target.csv"):
    """
    Loads the target CSV data which must include columns:
    time, close, and manual_signal.
    """
    df = pd.read_csv(csv_path)
    df['DateTime'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
    return df

# Load daily target data (used for training the weights)
df = load_target_data()

#######################################
# Backtesting and Plotting Functions
#######################################
def backtest_equity(df, binary_signal):
    """
    Simulate an equity curve.
    When signal == 1, assume full investment (compound returns);
    when 0, equity remains unchanged.
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
    Plot price chart with vertical lines at aggregated signal transitions and the equity curve.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10), sharex=True)
    
    # Plot price with signal markers
    ax1.plot(df.index, df['close'], label='Price', color='black', linewidth=1.5)
    transitions = np.where(np.diff(binary_signal) != 0)[0] + 1
    for idx in transitions:
        dt = df.index[idx]
        if binary_signal[idx] == 1:
            ax1.axvline(x=dt, color='green', linestyle='--', alpha=0.7)
        else:
            ax1.axvline(x=dt, color='red', linestyle='--', alpha=0.7)
    ax1.set_title(f"{title_prefix} Signals on Price Chart")
    ax1.set_ylabel("Price")
    ax1.legend()
    
    # Plot equity curve
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
# (These functions load their optimal settings from JSON files.)

def get_hull_suite_signal(df):
    settings = load_json_settings("./MTPI/settings/hull_suite_settings.json")
    source = settings.get("source", "close")
    length = int(settings.get("length", 55))
    def wma(series, window):
        weights = np.arange(1, window+1)
        return series.rolling(window, min_periods=1).apply(lambda x: np.dot(x, weights[:len(x)])/weights[:len(x)].sum(), raw=True)
    half_length = int(round(length/2))
    sqrt_length = int(round(np.sqrt(length)))
    hma = wma(2 * wma(df[source], half_length) - wma(df[source], length), sqrt_length)
    return np.array((hma > hma.shift(2)).astype(int), dtype=int)

def get_qtrend_signal(df):
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
    sig = np.where(close > m + epsilon, 1, np.where(close < m - epsilon, 0, np.nan))
    sig = pd.Series(sig, index=df.index).ffill().fillna(0).astype(int)
    return np.array(sig, dtype=int)

def get_smieo_signal(df):
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
        xEMA_SMI = xSMI.ewm(span=SmthLen, adjust=False).mean()  # computed for reference
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

def get_trend_follower_signal(df):
    settings = load_json_settings("./MTPI/settings/trend_follower_settings.json")
    prd = int(settings.get("prd", 20))
    maprd = int(settings.get("maprd", 20))
    rateinp = float(settings.get("rateinp", 1))
    linprd = int(settings.get("linprd", 5))
    matype = settings.get("matype", "EMA")
    ulinreg = settings.get("ulinreg", True)
    
    # Local implementations
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
# 6. Aggregated Signal and Weight Optimization
#######################################
def aggregate_weighted_signal(w1, w2, w3, w4, signals):
    """
    Given four binary signal arrays (each of shape (n,)), compute the weighted aggregate.
    First, normalize weights so they sum to 1, then compute:
    
        aggregated = w1*hull + w2*qtrend + w3*smieo + w4*trend_follower
    
    Finally, threshold the aggregated signal at 0.5 to produce a final binary signal.
    """
    weights = np.array([w1, w2, w3, w4])
    if np.sum(weights) == 0:
        norm_weights = np.ones_like(weights)/len(weights)
    else:
        norm_weights = weights / np.sum(weights)
    aggregated_cont = (norm_weights[0]*signals[0] +
                        norm_weights[1]*signals[1] +
                        norm_weights[2]*signals[2] +
                        norm_weights[3]*signals[3])
    # Final binary signal: 1 if aggregated average >= 0.5, else 0.
    final_signal = (aggregated_cont >= 0.5).astype(int)
    return final_signal

def objective_weights(w1, w2, w3, w4):
    """
    Objective function to train the weights.
    It computes the aggregated binary signal from the four indicator signals,
    then computes the mean absolute error (MAE) between the aggregated signal and the target ISP.
    Returns the negative MAE (to maximize the match).
    """
    # Get individual signals (they are binary arrays, same length as df)
    hull = get_hull_suite_signal(df)
    qtrend = get_qtrend_signal(df)
    smieo = get_smieo_signal(df)
    trend = get_trend_follower_signal(df)
    
    signals = [hull, qtrend, smieo, trend]
    aggregated = aggregate_weighted_signal(w1, w2, w3, w4, signals)
    
    target = df['manual_signal'].values.astype(int)
    mae = np.mean(np.abs(aggregated - target))
    return -mae

#######################################
# 7. Optimize Weights Using Bayesian Optimization
#######################################
pbounds_weights = {
    'w1': (0, 1),
    'w2': (0, 1),
    'w3': (0, 1),
    'w4': (0, 1)
}

optimizer_weights = BayesianOptimization(
    f=objective_weights,
    pbounds=pbounds_weights,
    random_state=42,
    verbose=2
)

optimizer_weights.maximize(init_points=5, n_iter=50)

best_weights = optimizer_weights.max['params']
print("Optimal Weights:")
print(best_weights)
print("Best Objective (negative MAE):", optimizer_weights.max['target'])

#######################################
# 8. Compute Final Aggregated Signal and Backtest Equity
#######################################
# Get individual signals
hull = get_hull_suite_signal(df)
qtrend = get_qtrend_signal(df)
smieo = get_smieo_signal(df)
trend = get_trend_follower_signal(df)
signals = [hull, qtrend, smieo, trend]

# Compute aggregated binary signal using the optimal weights
w1_opt = best_weights['w1']
w2_opt = best_weights['w2']
w3_opt = best_weights['w3']
w4_opt = best_weights['w4']

aggregated_signal = aggregate_weighted_signal(w1_opt, w2_opt, w3_opt, w4_opt, signals)

# Backtest equity using the aggregated signal
aggregated_equity = backtest_equity(df, aggregated_signal)

#######################################
# 9. Plot the Aggregated Results
#######################################
plot_backtest(df, aggregated_signal, aggregated_equity, title_prefix="Aggregated Weighted Indicator")
print(aggregated_equity[-1])