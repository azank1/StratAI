import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import os, json
import matplotlib.pyplot as plt

###############################
# Q-Trend Indicator Functions
###############################
def compute_atr(df, atr_period):
    """
    Compute the Average True Range (ATR) as a rolling mean of the True Range,
    shifted by one bar.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean().shift(1)
    return atr

def compute_qtrend_signal(df, p=200, atr_p=14, mult=1.0, mode="Type A",
                          source="close", use_ema_smoother=False, src_ema_period=3):
    """
    Computes the Q-Trend indicator signal.
    
    Parameters:
      - df: DataFrame with columns: 'open', 'high', 'low', 'close'
      - p: Trend period (number of bars lookback for highest/lowest)
      - atr_p: ATR period
      - mult: ATR multiplier (sensitivity measure)
      - mode: "Type A" or "Type B" (affects crossover logic)
      - source: Which price column to use (e.g. "close", "open", "high", "low")
      - use_ema_smoother: Boolean flag to smooth the source price with EMA
      - src_ema_period: EMA period if smoothing is applied
      
    Returns:
      - final_signal: numpy array of binary signals (1 for buy, 0 for sell)
    """
    src = df[source].copy()
    if use_ema_smoother:
        src = src.ewm(span=src_ema_period, adjust=False).mean()
    
    h = src.rolling(window=p, min_periods=1).max()
    l = src.rolling(window=p, min_periods=1).min()
    m = (h + l) / 2.0  # initial trend line
    
    atr = compute_atr(df, atr_p)
    epsilon = mult * atr  # sensitivity threshold
    
    n = len(df)
    final_signal = np.zeros(n, dtype=int)
    ls = None  # last signal ("B" for buy, "S" for sell)
    m_vals = m.copy()
    
    for i in range(1, n):
        price_prev = src.iloc[i-1]
        price_curr = src.iloc[i]
        m_prev = m_vals.iloc[i-1]
        eps_prev = epsilon.iloc[i-1] if not np.isnan(epsilon.iloc[i-1]) else 0
        eps_curr = epsilon.iloc[i] if not np.isnan(epsilon.iloc[i]) else 0
        
        if mode == "Type B":
            change_up = ((price_prev < m_prev + eps_prev and price_curr >= m_prev + eps_prev) or (price_curr > m_prev + eps_prev))
            change_down = ((price_prev > m_prev - eps_prev and price_curr <= m_prev - eps_prev) or (price_curr < m_prev - eps_prev))
        else:  # Type A
            change_up = ((price_prev <= m_prev + eps_prev and price_curr > m_prev + eps_curr) or (price_curr > m_prev + eps_curr))
            change_down = ((price_prev >= m_prev - eps_prev and price_curr < m_prev - eps_curr) or (price_curr < m_prev - eps_curr))
        
        if change_up:
            m_vals.iloc[i] = m_prev + eps_curr
            ls = "B"
        elif change_down:
            m_vals.iloc[i] = m_prev - eps_curr
            ls = "S"
        else:
            m_vals.iloc[i] = m_prev
        
        if ls is None:
            final_signal[i] = 0
        else:
            final_signal[i] = 1 if ls == "B" else 0

    return final_signal

###############################
# Bayesian Optimization for Q-Trend
###############################
def objective_qtrend(p, atr_p, mult, mode_idx, source_idx, use_ema_smoother_idx, src_ema_period):
    p = int(round(p))
    atr_p = int(round(atr_p))
    src_ema_period = int(round(src_ema_period))
    
    mode = "Type A" if mode_idx < 0.5 else "Type B"
    SOURCES = ['close', 'open', 'high', 'low']
    source_idx = int(round(source_idx))
    source_idx = max(0, min(source_idx, len(SOURCES)-1))
    source = SOURCES[source_idx]
    
    use_ema_smoother = True if use_ema_smoother_idx >= 0.5 else False
    
    signal = compute_qtrend_signal(df, p=p, atr_p=atr_p, mult=mult, mode=mode,
                                   source=source, use_ema_smoother=use_ema_smoother,
                                   src_ema_period=src_ema_period)
    signal = np.array(signal, dtype=float)
    target = df['manual_signal'].values
    mae = np.mean(np.abs(signal - target))
    
    # Penalize frequent signal transitions to favor smoother signals.
    transitions = np.sum(np.abs(np.diff(signal)))
    transition_rate = transitions / len(signal)
    lambda_penalty = 0.1  # adjust penalty weight as needed
    
    total_loss = mae + lambda_penalty * transition_rate
    return -total_loss  # negative for maximization

###############################
# Load CSV Data and Preprocess
###############################
csv_file = "./CSVdata/target.csv"
df = pd.read_csv(csv_file)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.sort_values('time', inplace=True)
df.reset_index(drop=True, inplace=True)
df['manual_signal'] = df['manual_signal'].ffill().fillna(0).astype(float)

###############################
# Set Parameter Bounds for Optimization
###############################
pbounds = {
    'p': (50, 400),
    'atr_p': (5, 50),
    'mult': (0.5, 2.0),
    'mode_idx': (0, 1),
    'source_idx': (0, 3),
    'use_ema_smoother_idx': (0, 1),
    'src_ema_period': (1, 10)
}

optimizer = BayesianOptimization(
    f=objective_qtrend,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=20)

best_params = optimizer.max['params']
best_loss = -optimizer.max['target']
print("Optimal Q-Trend Parameters:")
print("Trend period (p):", int(round(best_params['p'])))
print("ATR period (atr_p):", int(round(best_params['atr_p'])))
print("ATR multiplier (mult):", best_params['mult'])
print("Signal mode:", "Type A" if best_params['mode_idx'] < 0.5 else "Type B")
print("Source:", ['close','open','high','low'][int(round(best_params['source_idx']))])
print("Use EMA smoother:", True if best_params['use_ema_smoother_idx'] >= 0.5 else False)
print("EMA smoother period (src_ema_period):", int(round(best_params['src_ema_period'])))
print("Best MAE:", best_loss)

###############################
# Save Optimal Settings to JSON
###############################
best_settings = {
    "p": int(round(best_params['p'])),
    "atr_p": int(round(best_params['atr_p'])),
    "mult": best_params['mult'],
    "mode": "Type A" if best_params['mode_idx'] < 0.5 else "Type B",
    "source": ['close','open','high','low'][int(round(best_params['source_idx']))],
    "use_ema_smoother": True if best_params['use_ema_smoother_idx'] >= 0.5 else False,
    "src_ema_period": int(round(best_params['src_ema_period']))
}
with open(os.path.join("MTPI", "./settings/qtrend_settings.json"), "w") as f:
    json.dump(best_settings, f, indent=4)

###############################
# Compute Equity Curve Based on Q-Trend Signal
###############################
final_signal = compute_qtrend_signal(df, 
                                     p=best_settings["p"],
                                     atr_p=best_settings["atr_p"],
                                     mult=best_settings["mult"],
                                     mode=best_settings["mode"],
                                     source=best_settings["source"],
                                     use_ema_smoother=best_settings["use_ema_smoother"],
                                     src_ema_period=best_settings["src_ema_period"])
final_signal = np.array(final_signal, dtype=float)

close_prices = df['close'].astype(float).values
equity = [1.0]
for i in range(1, len(close_prices)):
    if final_signal[i] == 1:
        equity.append(equity[-1] * (close_prices[i] / close_prices[i-1]))
    else:
        equity.append(equity[-1])
equity = np.array(equity)

###############################
# Plot BTC Price with Vertical Signal Labels and Equity Curve
###############################
# Create a figure with two subplots: top for BTC price with signals, bottom for equity curve.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10), sharex=True)

# Plot BTC price on top subplot
ax1.plot(df['time'], close_prices, label='BTC Price', color='black', linewidth=1.5)
# Plot ISP (the target manual_signal) as a blue dashed line for reference.
ax1.plot(df['time'], df['manual_signal'], label='ISP (Manual Signal)', color='blue', linestyle='--', alpha=0.7)

# Determine signal transitions in the final_signal array.
transitions = np.where(np.diff(final_signal) != 0)[0] + 1
for idx in transitions:
    dt = df['time'].iloc[idx]
    # Entry signal: when final_signal changes to 1.
    if final_signal[idx] == 1:
        ax1.axvline(x=dt, color='green', linestyle='--', alpha=0.7,
                    label='Entry Signal' if 'Entry Signal' not in [l.get_label() for l in ax1.get_lines()] else "")
    # Exit signal: when final_signal changes to 0.
    else:
        ax1.axvline(x=dt, color='red', linestyle='--', alpha=0.7,
                    label='Exit Signal' if 'Exit Signal' not in [l.get_label() for l in ax1.get_lines()] else "")

ax1.set_title("BTC Price with Q-Trend Indicator Signals (TCT) and ISP")
ax1.set_ylabel("Price")
ax1.legend()

# Plot the equity curve on the bottom subplot.
ax2.plot(df['time'], equity, label='Equity Curve', color='orange', linewidth=2)
ax2.set_title("Equity Curve Based on Q-Trend Signal")
ax2.set_xlabel("Date")
ax2.set_ylabel("Equity")
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
