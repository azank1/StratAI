import pandas as pd
import numpy as np

# Helper function: Weighted Moving Average
def wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Hull Suite indicator variants
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

# Hull Suite hypothesis function: computes a binary signal based on chosen parameters
def compute_hull_suite_signal(df, source_col, mode, length, length_mult):
    eff_length = int(round(length * length_mult))
    src = df[source_col]
    
    if mode == "Hma":
        hull = HMA(src, eff_length)
    elif mode == "Ehma":
        hull = EHMA(src, eff_length)
    elif mode == "Thma":
        hull = THMA(src, eff_length / 2)
    else:
        raise ValueError("Invalid mode. Choose from 'Hma', 'Ehma', or 'Thma'.")
    
    hull_shifted = hull.shift(2)
    # Generate binary signal: 1 if MHULL > SHULL, else 0.
    predicted_signal = np.where(hull > hull_shifted, 1, 0)
    return predicted_signal

# Loss function: Mean Absolute Error between predicted and target signals
def loss_function(predicted, target):
    return np.mean(np.abs(predicted - target))

# Load CSV data
df = pd.read_csv("./CSVdata/BTC.csv")
# Assume CSV has at least: 'time', 'open', 'high', 'low', 'close', and 'manual_signal'
# Forward-fill the manual_signal so every candle gets a valid target signal.
df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
target = df['manual_signal'].values

# Define parameter ranges for grid search
modes = ["Hma", "Ehma", "Thma"]
sources = ["close", "open", "high", "low"]
length_range = np.arange(20, 101, 5)      # e.g., from 20 to 100 in steps of 5
mult_range = np.arange(0.5, 2.1, 0.1)       # e.g., from 0.5 to 2.0 in steps of 0.1

best_loss = np.inf
best_params = None

# Grid search over all combinations
for mode in modes:
    for source in sources:
        for length in length_range:
            for mult in mult_range:
                predicted = compute_hull_suite_signal(df, source_col=source, mode=mode, length=length, length_mult=mult)
                # Replace NaN values (from shift) with 0.
                predicted = np.nan_to_num(predicted, nan=0)
                current_loss = loss_function(predicted, target)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = (mode, source, length, mult)

print("Best parameters (mode, source, length, length_mult):", best_params)
print("Loss with best parameters:", best_loss)
