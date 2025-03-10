import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization

# 1) Define your categorical options.
MODES = ["Hma", "Ehma", "Thma"]
SOURCES = ["close", "open", "high", "low"]

# 2) Weighted moving average helper.
def wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

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

# 4) Hull Suite hypothesis function:
def compute_hull_suite_signal(df, source_col, mode, length, length_mult):
    """
    mode: one of 'Hma', 'Ehma', 'Thma'
    source_col: e.g. 'close', 'open', 'high', 'low'
    length, length_mult: numeric
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
    predicted_signal = np.where(hull > hull_shifted, 1, 0)
    return predicted_signal

# 5) Loss function: Mean Absolute Error
def loss_function(predicted, target):
    return np.mean(np.abs(predicted - target))

# 6) Load and prepare your CSV data
df = pd.read_csv("target.csv")
# Assume columns: 'time', 'open', 'high', 'low', 'close', 'manual_signal'
df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
target = df['manual_signal'].values

# 7) Define the objective function for Bayesian optimization.
def objective(length, length_mult, mode_idx, source_idx):
    """
    length, length_mult: continuous hyperparameters
    mode_idx, source_idx: continuous, but we will map them to integers
    """
    # Round to nearest integer to pick a valid index in MODES and SOURCES
    mode_idx = int(round(mode_idx))
    source_idx = int(round(source_idx))
    
    # Clip in case rounding goes out of bounds
    mode_idx = max(0, min(mode_idx, len(MODES) - 1))
    source_idx = max(0, min(source_idx, len(SOURCES) - 1))
    
    mode = MODES[mode_idx]
    source = SOURCES[source_idx]
    
    # Convert length to an integer
    length = int(round(length))
    
    predicted = compute_hull_suite_signal(df, source_col=source, mode=mode, length=length, length_mult=length_mult)
    predicted = np.nan_to_num(predicted, nan=0)
    current_loss = loss_function(predicted, target)
    # Bayesian optimization tries to maximize the objective, so return negative loss
    return -current_loss

# 8) Define parameter bounds (continuous ranges).
# For mode_idx: [0, len(MODES)-1], for source_idx: [0, len(SOURCES)-1]
pbounds = {
    'length': (20, 100),
    'length_mult': (0.5, 2.0),
    'mode_idx': (0, len(MODES) - 1),
    'source_idx': (0, len(SOURCES) - 1),
}

# 9) Initialize Bayesian Optimizer
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# 10) Run the optimizer
optimizer.maximize(init_points=5, n_iter=20)

# 11) Retrieve the best parameters found
best_params = optimizer.max['params']
best_loss = -optimizer.max['target']  # we returned negative loss, so invert
print("Best param set:", best_params)
print("Best loss:", best_loss)

# 12) Map the best mode_idx/source_idx to actual strings
best_mode = MODES[int(round(best_params['mode_idx']))]
best_source = SOURCES[int(round(best_params['source_idx']))]
print("Best mode:", best_mode)
print("Best source:", best_source)
