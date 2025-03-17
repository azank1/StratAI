import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import os, json
import matplotlib.pyplot as plt

#######################################
# 1. Load and Merge Data
#######################################
# Load daily target CSV with daily ISP (columns: time, manual_signal)
daily_csv = "./CSVdata/target.csv"  # Daily data with desired target signal
daily_df = pd.read_csv(daily_csv)
daily_df['time'] = pd.to_datetime(daily_df['time'], unit='s')
daily_df.sort_values('time', inplace=True)
daily_df['date'] = daily_df['time'].dt.date

# Load intraday (2D) price data CSV (columns: time, open, high, low, close)
chart_csv = "./CSVdata/target.csv"
chart_df = pd.read_csv(chart_csv)
chart_df['time'] = pd.to_datetime(chart_df['time'], unit='s')
chart_df.sort_values('time', inplace=True)

# Merge the daily target into intraday data so each intraday bar gets the most recent daily target.
merged_df = pd.merge_asof(chart_df, daily_df[['time', 'manual_signal', 'date']], on='time', direction='backward')
merged_df['date'] = merged_df['time'].dt.date

#######################################
# 2. SMIEO Indicator Functions (Using xSMI Only)
#######################################
def compute_smi_ergodic_indicator(df, fastPeriod=4, slowPeriod=8, SmthLen=3, source="close"):
    """
    Computes the continuous SMI Ergodic Oscillator.
    
    Calculation:
      - xPrice = df[source]
      - xPrice1 = xPrice - xPrice.shift(1)
      - xPrice2 = abs(xPrice - xPrice.shift(1))
      - xSMA_R = EMA(EMA(xPrice1, fastPeriod), slowPeriod)
      - xSMA_aR = EMA(EMA(xPrice2, fastPeriod), slowPeriod)
      - xSMI = xSMA_R / xSMA_aR (NaNs replaced with 0)
      - xEMA_SMI = EMA(xSMI, SmthLen)  [computed for reference only]
      
    Returns:
      (xSMI, xEMA_SMI) as pandas Series.
    """
    xPrice = df[source].copy()
    xPrice1 = xPrice - xPrice.shift(1)
    xPrice2 = xPrice.diff().abs()
    
    xSMA_R = xPrice1.ewm(span=fastPeriod, adjust=False).mean().ewm(span=slowPeriod, adjust=False).mean()
    xSMA_aR = xPrice2.ewm(span=fastPeriod, adjust=False).mean().ewm(span=slowPeriod, adjust=False).mean()
    
    xSMI = (xSMA_R / xSMA_aR).fillna(0)
    xEMA_SMI = xSMI.ewm(span=SmthLen, adjust=False).mean()
    return xSMI, xEMA_SMI

def compute_smi_ergodic_signal(df, fastPeriod=4, slowPeriod=8, SmthLen=3, thresh=0.0, source="close"):
    """
    Computes a binary SMIEO signal using only the oscillator (xSMI).
    
    Signal rule:
      - If xSMI > thresh, signal = 1 (entry).
      - If xSMI < -thresh, signal = 0 (exit).
      - Otherwise, hold the previous signal.
      
    (Here, thresh can be used as a dead-zone parameter to ignore very small oscillations.)
    
    Returns:
      final_signal: numpy array of binary signals.
    """
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
    return final_signal

#######################################
# 3. Aggregation Function: Intraday to Daily Signal
#######################################
def aggregate_intraday_signal(df, smooth_window):
    """
    Aggregates the intraday binary signal to a daily signal.
    For each day, take the last intraday signal and then apply a rolling mean with window=smooth_window.
    The final daily signal is 1 if the smoothed value >= 0.5, else 0.
    
    Returns:
      daily_signal: DataFrame with columns 'date' and 'final_signal'
    """
    daily_signal = df.groupby('date')['signal'].last().reset_index()
    daily_signal['smoothed'] = daily_signal['signal'].rolling(window=int(smooth_window), min_periods=1).mean()
    daily_signal['final_signal'] = (daily_signal['smoothed'] >= 0.5).astype(int)
    return daily_signal

#######################################
# 4. Objective Function for Training SMIEO on 2D Data Using Daily Target
#######################################
def objective_smi(fastPeriod, slowPeriod, SmthLen, thresh, smooth_window):
    """
    Objective function for Bayesian optimization.
    
    1. Compute the intraday SMIEO binary signal on the merged 2D data using the given parameters.
    2. Save the signal in the merged DataFrame.
    3. Aggregate the intraday signal to a daily signal using a rolling average (smoothing over smooth_window days).
    4. Compute the mean absolute error (MAE) between the aggregated daily signal and the daily target (manual_signal).
    5. Additionally, penalize excessive daily transitions to encourage a smoother (less trade‐heavy) signal.
    6. Return the negative total loss (so that the optimizer maximizes the match).
    """
    fastPeriod = int(round(fastPeriod))
    slowPeriod = int(round(slowPeriod))
    SmthLen = int(round(SmthLen))
    thresh = float(thresh)
    smooth_window = int(round(smooth_window))
    source = "close"
    
    # Compute intraday binary signal using xSMI only.
    signal = compute_smi_ergodic_signal(merged_df, fastPeriod=fastPeriod, slowPeriod=slowPeriod,
                                        SmthLen=SmthLen, thresh=thresh, source=source)
    merged_df['signal'] = signal
    
    # Aggregate the intraday signal to a daily signal.
    daily_agg = aggregate_intraday_signal(merged_df, smooth_window)
    # Merge with daily target data using 'date'
    daily_merge = pd.merge(daily_agg, daily_df[['date', 'manual_signal']], on='date', how='inner')
    
    mae = np.mean(np.abs(daily_merge['final_signal'] - daily_merge['manual_signal']))
    
    # Penalize the number of daily transitions to encourage smoothness.
    daily_transitions = np.sum(np.abs(np.diff(daily_merge['final_signal'])))
    trade_penalty = 0.5 * daily_transitions  # adjust the penalty factor as needed
    
    total_loss = mae + trade_penalty
    return -total_loss  # Negative, since we want to maximize the match

#######################################
# 5. Training via Bayesian Optimization
#######################################
# Use a larger smoothing window to encourage fewer daily trade transitions.
# We now set smooth_window bounds to be higher (e.g., from 10 to 50 days).
pbounds = {
    'fastPeriod': (1, 10),
    'slowPeriod': (1, 20),
    'SmthLen': (1, 10),
    'thresh': (0.0, 0.2),         # Use a smaller threshold since xSMI oscillates around 0.
    'smooth_window': (10, 50)     # Larger smoothing window for daily aggregation.
}

optimizer = BayesianOptimization(
    f=objective_smi,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Train for at least 55 epochs (here: 5 init points + 50 iterations)
optimizer.maximize(init_points=5, n_iter=50)

best_params = optimizer.max['params']
final_mae = -optimizer.max['target']
print("Optimal SMIEO Training Parameters:")
print("fastPeriod:", int(round(best_params['fastPeriod'])))
print("slowPeriod:", int(round(best_params['slowPeriod'])))
print("SmthLen:", int(round(best_params['SmthLen'])))
print("thresh:", best_params['thresh'])
print("smooth_window:", int(round(best_params['smooth_window'])))
print("Final MAE on Daily Signal:", final_mae)

# Save optimal settings
with open(os.path.join("MTPI", "./settings/smi_ergodic_settings.json"), "w") as f:
    json.dump(best_params, f, indent=4)

#######################################
# 6. Compute Final SMIEO Signal on Merged 2D Data & Simulate Equity Curve
#######################################
final_signal = compute_smi_ergodic_signal(merged_df, 
                                          fastPeriod=int(round(best_params['fastPeriod'])),
                                          slowPeriod=int(round(best_params['slowPeriod'])),
                                          SmthLen=int(round(best_params['SmthLen'])),
                                          thresh=best_params['thresh'],
                                          source=best_params.get("source", "close"))
final_signal = np.array(final_signal, dtype=float)

close_prices = merged_df['close'].astype(float).values
equity = [1.0]
for i in range(1, len(close_prices)):
    if final_signal[i] == 1:
        equity.append(equity[-1] * (close_prices[i] / close_prices[i-1]))
    else:
        equity.append(equity[-1])
equity = np.array(equity)

#######################################
# 7. Plot the Results
#######################################
fig, axs = plt.subplots(3, 1, figsize=(14,12), sharex=True)

# Subplot 1: Intraday BTC Price with SMIEO Signal Markers
axs[0].plot(merged_df['time'], close_prices, label='BTC Price', color='black', linewidth=1.5)
transitions = np.where(np.diff(final_signal) != 0)[0] + 1
for idx in transitions:
    dt = merged_df['time'].iloc[idx]
    if final_signal[idx] == 1:
        axs[0].axvline(x=dt, color='green', linestyle='--', alpha=0.7)
    else:
        axs[0].axvline(x=dt, color='red', linestyle='--', alpha=0.7)
axs[0].set_title("Intraday BTC Price with SMIEO Indicator Signals")
axs[0].set_ylabel("Price")
axs[0].legend()

# Subplot 2: Continuous SMIEO Oscillator (xSMI)
xSMI, _ = compute_smi_ergodic_indicator(merged_df, 
                                        fastPeriod=int(round(best_params['fastPeriod'])),
                                        slowPeriod=int(round(best_params['slowPeriod'])),
                                        SmthLen=int(round(best_params['SmthLen'])),
                                        source="close")
axs[1].plot(merged_df['time'], xSMI, label='xSMI (Oscillator)', color='green', linewidth=1.5)
axs[1].axhline(0, color='gray', linestyle='--', label='Zero Line')
axs[1].set_title("SMIEO Oscillator (xSMI)")
axs[1].set_ylabel("Indicator Value")
axs[1].legend()

# Subplot 3: Equity Curve Based on Intraday Signal
axs[2].plot(merged_df['time'], equity, label='Equity Curve', color='orange', linewidth=2)
axs[2].set_title("Equity Curve Based on SMIEO Signal")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Equity")
axs[2].legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(equity[-1])