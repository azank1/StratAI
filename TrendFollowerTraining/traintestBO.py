import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import os, json
import matplotlib.pyplot as plt

#######################################
# Global Options for Trend Follower
#######################################
MATYPE_OPTIONS = ['EMA', 'SMA', 'RMA', 'WMA', 'VWMA']
ULINREG_OPTIONS = [False, True]

#######################################
# 1. Load Target CSV Data
#######################################
# This CSV contains daily data with columns: time, close, manual_signal.
df = pd.read_csv("./CSVdata/target.csv")
df['DateTime'] = pd.to_datetime(df['time'], unit='s')
df.sort_values('DateTime', inplace=True)
df.set_index('DateTime', inplace=True)
df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
target = df['manual_signal'].values
# For aggregation we also create a date column (daily)
df['date'] = df.index.date

#######################################
# 2. Trend Follower Indicator Functions (Original Pinescript Logic)
#######################################
def wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window, min_periods=1).apply(
        lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True
    )

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
        return series.ewm(span=period, adjust=False).mean()  # approximation for RMA
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

def compute_trend_follower(df, 
                           matype='EMA', 
                           prd=20,        
                           maprd=20,      
                           rateinp=1,     
                           ulinreg=True,  
                           linprd=5):
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
    _ret = trend * diff / safe_chan
    _ret = _ret.fillna(0)
    return _ret

#######################################
# 3. Binary Signal Conversion & Smoothing
#######################################
def binary_signal_from_indicator(indicator):
    """
    Converts the continuous Trend Follower indicator to a binary signal.
    Signal = 1 if indicator > 0, else 0.
    """
    return (indicator > 0).astype(int)

def smooth_binary_signal(binary_series, smooth_window):
    """
    Smooths a binary signal by applying a rolling mean over 'smooth_window' periods,
    then thresholding at 0.5.
    Returns a pandas Series of the aggregated binary signal.
    """
    smoothed = binary_series.rolling(window=smooth_window, min_periods=1).mean()
    return (smoothed >= 0.5).astype(int)

#######################################
# 4. Objective Function for Training Trend Follower Indicator
#######################################
def objective_trend(prd, maprd, rateinp, linprd, smooth_window, matype_idx, ulinreg_idx):
    """
    Objective function that trains the Trend Follower indicator on the daily target CSV.
    
    Steps:
      1. Compute the continuous Trend Follower indicator using the given parameters.
      2. Convert it to a binary signal (1 if indicator > 0, else 0).
      3. Smooth the binary signal using a rolling mean with window 'smooth_window'
         to favor longer entries/exits.
      4. Compute the mean absolute error (MAE) between the smoothed signal and the target (manual_signal).
      5. Additionally, add a penalty proportional to the number of transitions in the smoothed signal,
         to encourage fewer, longer trades.
      6. Return the negative total loss (for maximization).
    """
    # Convert parameters to proper types
    prd = int(round(prd))
    maprd = int(round(maprd))
    linprd = int(round(linprd))
    smooth_window = int(round(smooth_window))
    rateinp = float(rateinp)
    
    matype_idx = int(round(matype_idx))
    matype_idx = max(0, min(matype_idx, len(MATYPE_OPTIONS) - 1))
    ulinreg_idx = int(round(ulinreg_idx))
    ulinreg_idx = max(0, min(ulinreg_idx, len(ULINREG_OPTIONS) - 1))
    
    matype = MATYPE_OPTIONS[matype_idx]
    ulinreg = ULINREG_OPTIONS[ulinreg_idx]
    
    # Compute the continuous indicator
    indicator = compute_trend_follower(df, matype=matype, prd=prd, maprd=maprd, 
                                        rateinp=rateinp, ulinreg=ulinreg, linprd=linprd)
    # Convert to binary signal: 1 if > 0, else 0.
    binary = binary_signal_from_indicator(indicator)
    # Apply smoothing to encourage longer positions
    smoothed = smooth_binary_signal(binary, smooth_window)
    
    # Compute MAE between smoothed signal and target ISP
    mae = np.mean(np.abs(smoothed.values - target))
    
    # Count daily transitions (changes in the smoothed signal)
    transitions = np.sum(np.abs(np.diff(smoothed.values)))
    # Penalize frequent transitions (the higher the transitions, the larger the penalty)
    lambda_penalty = 0.1  # Adjust this penalty factor as needed
    penalty = lambda_penalty * transitions / len(smoothed)
    
    total_loss = mae + penalty
    return -total_loss  # Negative for maximization

#######################################
# 5. Bayesian Optimization for Training
#######################################
pbounds = {
    'prd': (10, 50),
    'maprd': (5, 50),
    'rateinp': (0.1, 3),
    'linprd': (2, 20),
    'smooth_window': (1, 20),
    'matype_idx': (0, len(MATYPE_OPTIONS)-1),
    'ulinreg_idx': (0, 1)
}

optimizer = BayesianOptimization(
    f=objective_trend,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Train over ~55 epochs (5 initial random points + 50 iterations)
optimizer.maximize(init_points=5, n_iter=50)

best_params = optimizer.max['params']
final_loss = -optimizer.max['target']
print("Optimal Trend Follower Parameters:")
print("prd:", int(round(best_params['prd'])))
print("maprd:", int(round(best_params['maprd'])))
print("rateinp:", best_params['rateinp'])
print("linprd:", int(round(best_params['linprd'])))
print("smooth_window:", int(round(best_params['smooth_window'])))
print("matype:", MATYPE_OPTIONS[int(round(best_params['matype_idx']))])
print("ulinreg:", ULINREG_OPTIONS[int(round(best_params['ulinreg_idx']))])
print("Final MAE+Penalty:", final_loss)

# Save optimal settings to JSON
with open(os.path.join("MTPI", "./settings/trend_follower_settings.json"), "w") as f:
    json.dump(best_params, f, indent=4)

#######################################
# 6. Compute Final Indicator and Backtest Equity
#######################################
final_indicator = compute_trend_follower(df, 
                                           matype=MATYPE_OPTIONS[int(round(best_params['matype_idx']))],
                                           prd=int(round(best_params['prd'])),
                                           maprd=int(round(best_params['maprd'])),
                                           rateinp=best_params['rateinp'],
                                           ulinreg=ULINREG_OPTIONS[int(round(best_params['ulinreg_idx']))],
                                           linprd=int(round(best_params['linprd'])))
final_binary = binary_signal_from_indicator(final_indicator)
# Apply smoothing (rolling average) to get the final aggregated signal
final_smoothed = smooth_binary_signal(final_binary, int(round(best_params['smooth_window'])))
final_smoothed = final_smoothed.values

# Simulate equity curve (full investment when smoothed signal is 1)
close_prices = df['close'].astype(float).values
equity = [1.0]
for i in range(1, len(close_prices)):
    if final_smoothed[i] == 1:
        equity.append(equity[-1] * (close_prices[i] / close_prices[i-1]))
    else:
        equity.append(equity[-1])
equity = np.array(equity)

#######################################
# 7. Plotting Results
#######################################
fig, axs = plt.subplots(3, 1, figsize=(14,12), sharex=True)

# Subplot 1: BTC Price and Trend Follower Indicator
axs[0].plot(df.index, df['close'], label='BTC Price', color='black', linewidth=1.5)
# Also plot the continuous indicator
axs[0].plot(df.index, final_indicator, label='Trend Follower', color='blue', linewidth=1.5)
axs[0].set_title("BTC Price with Trend Follower Indicator")
axs[0].set_ylabel("Price / Indicator")
axs[0].legend()

# Subplot 2: Binary Signal & Smoothed Signal
axs[1].plot(df.index, final_binary, label='Raw Binary Signal', color='gray', linestyle=':', linewidth=1)
axs[1].plot(df.index, final_smoothed, label='Smoothed Signal', color='green', linewidth=1.5)
axs[1].axhline(0.5, color='red', linestyle='--', label='Threshold')
axs[1].set_title("Binary Signal and Smoothed Aggregated Signal")
axs[1].set_ylabel("Signal")
axs[1].legend()

# Subplot 3: Equity Curve Based on Final Signal
axs[2].plot(df.index, equity, label='Equity Curve', color='orange', linewidth=2)
axs[2].set_title("Equity Curve Based on Trend Follower Signal")
axs[2].set_xlabel("Date")
axs[2].set_ylabel("Equity")
axs[2].legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
