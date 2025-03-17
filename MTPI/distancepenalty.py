import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

########################################
# 1. Data Loading and Preprocessing
########################################
def load_data(csv_file):
    """
    Load price data from CSV.
    Expected columns: 'time', 'close', 'open', 'high', 'low', 'manual_signal'
    'time' is assumed to be Unix timestamps in seconds.
    """
    df = pd.read_csv(csv_file)
    df.columns = [col.lower() for col in df.columns]
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('time', inplace=True)
    df.set_index('time', inplace=True)
    # Forward-fill manual_signal target
    if 'manual_signal' in df.columns:
        df['manual_signal'] = df['manual_signal'].ffill().fillna(0).astype(int)
    else:
        raise ValueError("CSV must contain a 'manual_signal' column.")
    return df

data_file = "./CSVdata/target.csv"  # Update path as needed
df = load_data(data_file)
print("Data loaded. Time range:", df.index[0], "to", df.index[-1])

########################################
# 2. Moving Average Functions
########################################
def SMA(series, period):
    """Compute Simple Moving Average."""
    return pd.Series(series).rolling(window=int(period), min_periods=int(period)).mean().to_numpy()

def EMA(series, period):
    """Compute Exponential Moving Average."""
    return pd.Series(series).ewm(span=int(period), adjust=False).mean().to_numpy()

########################################
# 3. MACD Indicator Functions
########################################
def compute_macd(df, fast_length, slow_length, signal_length, sma_source, sma_signal, src='close'):
    """
    Compute MACD components:
      - fast_length: period for fast MA
      - slow_length: period for slow MA
      - signal_length: period for signal line smoothing
      - sma_source: "SMA" or "EMA" for oscillator MA type
      - sma_signal: "SMA" or "EMA" for signal line MA type
    Returns:
      (macd_line, signal_line, histogram) as numpy arrays.
    """
    price = df[src].values.astype(float)
    # Calculate fast and slow moving averages using the chosen method.
    if sma_source == "SMA":
        fast_ma = SMA(price, fast_length)
        slow_ma = SMA(price, slow_length)
    else:
        fast_ma = EMA(price, fast_length)
        slow_ma = EMA(price, slow_length)
    
    macd_line = fast_ma - slow_ma
    
    # Signal line calculation.
    if sma_signal == "SMA":
        signal_line = SMA(macd_line, signal_length)
    else:
        signal_line = EMA(macd_line, signal_length)
    
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def generate_macd_signal(df, fast_length, slow_length, signal_length, sma_source, sma_signal, src='close'):
    """
    Generate binary trading signals:
      - Signal = 1 (enter) if MACD line > Signal line, else 0 (exit).
    """
    macd_line, signal_line, _ = compute_macd(df, fast_length, slow_length, signal_length, sma_source, sma_signal, src)
    return np.where(macd_line > signal_line, 1, 0)

########################################
# 4. Signal Smoothing, Transition Penalty, and MAE Calculation
########################################
def smooth_signals(raw_signals, window=7):
    """
    Apply a rolling mean over the binary signals with the specified window,
    then threshold at 0.5 to produce final binary signals.
    Larger window => more smoothing => fewer flips.
    """
    smoothed = pd.Series(raw_signals).rolling(window=window, min_periods=1).mean()
    return smoothed.apply(lambda x: 1 if x > 0.5 else 0).to_numpy()

def transition_fraction(signals):
    """
    Calculate the fraction of signal changes over the total length.
    This is used to penalize excessive flips.
    """
    transitions = np.sum(np.abs(np.diff(signals)))
    frac = transitions / (len(signals) - 1) if len(signals) > 1 else 0
    return frac

########################################
# 5. Objective Function (ISP Matching + Noise Penalty)
########################################
def objective_macd(fast_length, slow_length, signal_length, sma_source_idx, sma_signal_idx,
                   penalty_flips=0.5):
    """
    Objective function for Bayesian optimization.
    1. Generates MACD binary signals using the provided parameters.
    2. Applies a rolling smoothing (window=7).
    3. Computes the mean absolute error (MAE) between the smoothed signals and the target ISP.
    4. Computes the fraction of transitions and penalizes it.
    5. Returns the negative of (MAE + penalty_flips * transitions).
    """
    # Convert parameters to integers.
    fast_length = int(round(fast_length))
    slow_length = int(round(slow_length))
    signal_length = int(round(signal_length))
    
    # Ensure fast_length < slow_length for valid MACD config.
    if fast_length >= slow_length:
        return -1000  # Large penalty for invalid config.
    
    options = ["SMA", "EMA"]
    sma_source = options[int(round(sma_source_idx))]
    sma_signal = options[int(round(sma_signal_idx))]
    
    raw_signals = generate_macd_signal(df, fast_length, slow_length, signal_length, sma_source, sma_signal, src='close')
    final_signals = smooth_signals(raw_signals, window=7)
    
    # Calculate MAE against the target ISP.
    target = df['manual_signal'].values
    n = min(len(final_signals), len(target))
    mae = np.mean(np.abs(final_signals[:n] - target[:n]))
    
    # Calculate fraction of transitions (noise).
    frac_flips = transition_fraction(final_signals)
    
    # Weighted sum of errors: MAE + penalty * fraction_of_flips
    total_error = mae + penalty_flips * frac_flips
    
    # Return negative for maximization
    return -total_error

########################################
# 6. Bayesian Optimization Setup for MACD
########################################
pbounds = {
    'fast_length': (5, 20),
    'slow_length': (21, 50),
    'signal_length': (5, 20),
    'sma_source_idx': (0, 1),  # 0: SMA, 1: EMA
    'sma_signal_idx': (0, 1)   # 0: SMA, 1: EMA
}

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=objective_macd,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

print("Starting Bayesian Optimization for MACD parameters (targeting ISP + noise penalty)...")
optimizer.maximize(init_points=5, n_iter=20)

best_params = optimizer.max['params']
best_error = -optimizer.max['target']  # This is the composite error (lower is better)

options = ["SMA", "EMA"]
print("\n=== Optimized MACD Parameters (Matching ISP, Less Noise) ===")
print(f"  Fast Length:   {int(round(best_params['fast_length']))}")
print(f"  Slow Length:   {int(round(best_params['slow_length']))}")
print(f"  Signal Length: {int(round(best_params['signal_length']))}")
print(f"  Oscillator MA: {options[int(round(best_params['sma_source_idx']))]}")
print(f"  Signal MA:     {options[int(round(best_params['sma_signal_idx']))]}")
print(f"Composite Error (MAE + NoisePenalty): {best_error:.4f}")

########################################
# 7. Final Signal Generation and Backtesting
########################################
final_raw_signals = generate_macd_signal(
    df,
    int(round(best_params['fast_length'])),
    int(round(best_params['slow_length'])),
    int(round(best_params['signal_length'])),
    options[int(round(best_params['sma_source_idx']))],
    options[int(round(best_params['sma_signal_idx']))],
    src='close'
)
final_signals = smooth_signals(final_raw_signals, window=7)

def backtest_strategy(df, signals, initial_equity=1.0):
    """Simple backtest: invest fully when signal=1; otherwise, hold cash."""
    prices = df['close'].values.astype(float)
    daily_returns = pd.Series(prices).pct_change().fillna(0).to_numpy()
    equity = [initial_equity]
    for i in range(1, len(daily_returns)):
        if signals[i] == 1:
            equity.append(equity[-1] * (1 + daily_returns[i]))
        else:
            equity.append(equity[-1])
    return pd.Series(equity, index=df.index)

equity_curve = backtest_strategy(df, final_signals, initial_equity=1.0)
final_equity = equity_curve.iloc[-1]
print(f"Final Equity from Backtest: {final_equity:.2f}")

########################################
# 8. Plotting the Results
########################################
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot Price with vertical lines for signal transitions.
ax1.plot(df.index, df['close'], label='BTC Price', color='black', linewidth=1.5)
transitions = np.where(np.diff(final_signals) != 0)[0] + 1
for idx in transitions:
    dt = df.index[idx]
    color_line = 'green' if final_signals[idx] == 1 else 'red'
    ax1.axvline(x=dt, color=color_line, linestyle='--', alpha=0.7)
ax1.set_title("BTC Price with Optimized MACD Signals (Less Noise)")
ax1.set_ylabel("Price")
ax1.legend()

# Plot Equity Curve.
ax2.plot(df.index, equity_curve, label='Equity Curve', color='orange', linewidth=2)
ax2.set_title("Backtested Equity Curve (MACD Strategy w/ Noise Penalty)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Equity")
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare generated vs. target ISP signals
plt.figure(figsize=(14, 4))
plt.plot(df.index, final_signals, label="Generated Signal", marker='o', linestyle='-', color='blue')
plt.plot(df.index, df['manual_signal'].values, label="Target ISP Signal", marker='x', linestyle='--', color='red')
plt.title("Comparison: Generated vs. Target ISP Signals (Less Noise)")
plt.xlabel("Date")
plt.ylabel("Signal (0 or 1)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
