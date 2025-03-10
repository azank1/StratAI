import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

###########################################
# Trend Follower Indicator Functions
###########################################

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
        return series.ewm(span=period, adjust=False).mean()  # Approximation for RMA
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

###########################################
# Bayesian Optimization Setup for Training
###########################################

# Define categorical options for matype and ulinreg
MATYPE_OPTIONS = ['EMA', 'SMA', 'RMA', 'WMA', 'VWMA']
ULINREG_OPTIONS = [False, True]  # 0 -> False, 1 -> True

def loss_function(predicted, target):
    return np.mean(np.abs(predicted - target))

# Load CSV data (target CSV)
df = pd.read_csv("./CSVdata/updated_target.csv")
df['DateTime'] = pd.to_datetime(df['time'], unit='s')
df.sort_values('DateTime', inplace=True)
df.set_index('DateTime', inplace=True)
df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
target = df['manual_signal'].values

def objective(prd, maprd, rateinp, linprd, matype_idx, ulinreg_idx):
    prd = int(round(prd))
    maprd = int(round(maprd))
    linprd = int(round(linprd))
    matype_idx = int(round(matype_idx))
    matype_idx = max(0, min(matype_idx, len(MATYPE_OPTIONS) - 1))
    ulinreg_idx = int(round(ulinreg_idx))
    ulinreg_idx = max(0, min(ulinreg_idx, len(ULINREG_OPTIONS) - 1))
    
    matype = MATYPE_OPTIONS[matype_idx]
    ulinreg = ULINREG_OPTIONS[ulinreg_idx]
    
    predicted = compute_trend_follower(df, 
                                       matype=matype,
                                       prd=prd,
                                       maprd=maprd,
                                       rateinp=rateinp,
                                       ulinreg=ulinreg,
                                       linprd=linprd)
    predicted = np.nan_to_num(predicted, nan=0)
    current_loss = loss_function(predicted, target)
    return -current_loss  # negative loss for maximization

# Parameter bounds (rateinp range adjusted to 0.1-3)
pbounds = {
    'prd': (10, 50),
    'maprd': (5, 50),
    'rateinp': (0.1, 3),
    'linprd': (2, 20),
    'matype_idx': (0, len(MATYPE_OPTIONS) - 1),
    'ulinreg_idx': (0, 1)
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

best_prd = int(round(best_params['prd']))
best_maprd = int(round(best_params['maprd']))
best_rateinp = best_params['rateinp']
best_linprd = int(round(best_params['linprd']))
best_matype = MATYPE_OPTIONS[int(round(best_params['matype_idx']))]
best_ulinreg = ULINREG_OPTIONS[int(round(best_params['ulinreg_idx']))]

print("Optimal Parameters:")
print("prd:", best_prd)
print("maprd:", best_maprd)
print("rateinp:", best_rateinp)
print("linprd:", best_linprd)
print("matype:", best_matype)
print("ulinreg:", best_ulinreg)
print("Best Loss (MAE):", best_loss)

###########################################
# Compute Indicator with Best Parameters
###########################################
trend_indicator = compute_trend_follower(
    df,
    matype=best_matype,
    prd=best_prd,
    maprd=best_maprd,
    rateinp=best_rateinp,
    ulinreg=best_ulinreg,
    linprd=best_linprd
)

# Create binary signal from the trend_indicator for equity calculation:
binary_signal = (trend_indicator > 0).astype(int)

# Compute equity curve: When in position (signal 1), equity multiplies by the candle return; else remains flat.
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

# Subplot 1: BTC Price with Trend Follower Indicator
ax1.plot(df.index, df['close'], label='BTC Price', color='black', linewidth=1.5)
ax1.plot(trend_indicator.index, trend_indicator, label='Trend Follower', color='blue', linewidth=1.5)
ax1.set_title("BTC Price with Trend Follower Indicator")
ax1.set_ylabel("Price / Trend Value")
ax1.legend(loc='upper left')

# Subplot 2: Equity Curve
ax2.plot(df.index, equity, label='Equity Curve', color='orange', linewidth=2)
ax2.set_title("Equity Curve Based on Trend Follower Signal")
ax2.set_xlabel("Date")
ax2.set_ylabel("Equity")
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
