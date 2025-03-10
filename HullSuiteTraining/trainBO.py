import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import os
import json

###########################################
# Hull Suite Indicator Functions
###########################################

# 1) Define categorical options.
MODES = ["Hma", "Ehma", "Thma"]
SOURCES = ["close", "open", "high", "low"]

# 2) Weighted moving average helper.
def wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window, min_periods=1).apply(
        lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True
    )

# 3) Hull Suite indicator variants.
def HMA(src, length):
    half_length = int(round(length / 2))
    sqrt_length = int(round(np.sqrt(length)))
    return wma(2 * wma(src, half_length) - wma(src, length), sqrt_length)

def EHMA(src, length):
    half_length = int(round(length / 2))
    sqrt_length = int(round(np.sqrt(length)))
    ema_half = src.ewm(span=half_length, adjust=False).mean()
    ema_full = src.ewm(span=length, adjust=False).mean()
    diff = 2 * ema_half - ema_full
    return diff.ewm(span=sqrt_length, adjust=False).mean()

def THMA(src, length):
    length = int(round(length))
    third_length = max(1, int(round(length / 3)))
    half_length = max(1, int(round(length / 2)))
    return wma(3 * wma(src, third_length) - wma(src, half_length) - wma(src, length), length)

# 4) Compute both MHULL and SHULL values from Hull Suite.
def compute_hull_suite_values(df, source_col, mode, length, length_mult):
    """
    mode: one of 'Hma', 'Ehma', 'Thma'
    source_col: e.g. 'close', 'open', 'high', 'low'
    length, length_mult: numeric
    Returns a tuple: (MHULL, SHULL) as pandas Series.
    """
    eff_length = int(round(length * length_mult))
    src = df[source_col]
    
    if mode == "Hma":
        hull = HMA(src, eff_length)
    elif mode == "Ehma":
        hull = EHMA(src, eff_length)
    elif mode == "Thma":
        hull = THMA(src, eff_length / 2)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    hull_shifted = hull.shift(2)
    return hull, hull_shifted

# 5) Compute binary signal from hull values.
def compute_hull_binary_signal(df, source_col, mode, length, length_mult):
    MHULL, SHULL = compute_hull_suite_values(df, source_col, mode, length, length_mult)
    binary_signal = np.where(MHULL > SHULL, 1, 0)
    return binary_signal

# 6) Loss function: Mean Absolute Error.
def loss_function(predicted, target):
    return np.mean(np.abs(predicted - target))

###########################################
# Bayesian Optimization Setup for Training Hull Suite
###########################################

# Load and prepare CSV data.
df = pd.read_csv("./CSVdata/updated_target.csv")
# Assume columns: 'time', 'open', 'high', 'low', 'close', 'manual_signal'
# Forward-fill the manual_signal column.
df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
target = df['manual_signal'].values

# Define the objective function for Bayesian optimization.
def objective(length, length_mult, mode_idx, source_idx):
    """
    length, length_mult: continuous hyperparameters.
    mode_idx, source_idx: continuous values mapped to integers.
    """
    mode_idx = int(round(mode_idx))
    source_idx = int(round(source_idx))
    mode_idx = max(0, min(mode_idx, len(MODES) - 1))
    source_idx = max(0, min(source_idx, len(SOURCES) - 1))
    
    mode = MODES[mode_idx]
    source = SOURCES[source_idx]
    
    length = int(round(length))
    
    predicted = compute_hull_binary_signal(df, source_col=source, mode=mode, length=length, length_mult=length_mult)
    predicted = np.nan_to_num(predicted, nan=0)
    current_loss = loss_function(predicted, target)
    return -current_loss  # negative loss for maximization

# Define parameter bounds.
pbounds = {
    'length': (20, 100),
    'length_mult': (0.5, 2.0),
    'mode_idx': (0, len(MODES) - 1),
    'source_idx': (0, len(SOURCES) - 1),
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=20)

best_params = optimizer.max['params']
best_loss = -optimizer.max['target']
print("Best parameter set:", best_params)
print("Best Loss (MAE):", best_loss)

best_mode = MODES[int(round(best_params['mode_idx']))]
best_source = SOURCES[int(round(best_params['source_idx']))]
best_length = int(round(best_params['length']))
best_length_mult = best_params['length_mult']
print("Best mode:", best_mode)
print("Best source:", best_source)

# Save the best parameters to a JSON file.

def save_settings(settings, filename):
    with open(os.path.join("./MTPI", filename), "w") as f:
        json.dump(settings, f, indent=4)

# Construct the settings dictionary for Hull Suite.
best_settings = {
    "mode": best_mode,
    "source": best_source,
    "length": best_length,
    "length_mult": best_length_mult
}

# Save these settings to a JSON file.
save_settings(best_settings, "./settings/hull_suite_settings.json")



###########################################
# Compute Hull Suite Indicator Values with Best Parameters
###########################################
MHULL, SHULL = compute_hull_suite_values(df, best_source, best_mode, best_length, best_length_mult)
MHULL = pd.Series(MHULL, index=df.index)
SHULL = pd.Series(SHULL, index=df.index)

# Compute binary signal for equity: if MHULL > SHULL, signal = 1; else 0.
binary_signal = (MHULL > SHULL).astype(int)

###########################################
# Equity Curve Calculation
###########################################
equity = [1.0]
close_prices = df['close'].values
for i in range(1, len(close_prices)):
    if binary_signal[i] == 1:
        equity.append(equity[-1] * (close_prices[i] / close_prices[i - 1]))
    else:
        equity.append(equity[-1])
equity = np.array(equity)

###########################################
# Plotting: One Figure with Two Subplots
###########################################
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Subplot 1: BTC Price with MHULL and SHULL Overlaid.
ax1.plot(df.index, df['close'], label='BTC Price', color='black', linewidth=1.5)
ax1.plot(MHULL.index, MHULL, label='MHULL', color='blue', linewidth=1.5)
ax1.plot(SHULL.index, SHULL, label='SHULL', color='red', linewidth=1.5)
ax1.set_title("BTC Price with Hull Suite Overlay (MHULL & SHULL)")
ax1.set_ylabel("Price / Indicator")
ax1.legend(loc='upper left')

# Subplot 2: Equity Curve Based on Hull Binary Signal.
ax2.plot(df.index, equity, label='Equity Curve', color='orange', linewidth=2)
ax2.set_title("Equity Curve Based on Hull Suite Signal")
ax2.set_xlabel("Date")
ax2.set_ylabel("Equity")
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
