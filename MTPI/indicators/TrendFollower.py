import numpy as np
import pandas as pd

def compute_trend_follower(df, 
                           matype='EMA', 
                           prd=20,        
                           maprd=20,      
                           rateinp=1,     
                           ulinreg=True,  
                           linprd=5):
    """
    Computes the Trend Follower indicator (continuous values), similar to your provided PineScript logic.
    Returns a pandas Series of trend values for each bar.
    """
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
        elif matype == 'RMA':
            # Approximate RMA with EMA
            return series.ewm(span=period, adjust=False).mean()
        elif matype == 'VWMA':
            if df is not None and 'volume' in df.columns:
                return (series * df['volume']).rolling(window=period, min_periods=1).sum() / \
                       df['volume'].rolling(window=period, min_periods=1).sum()
            else:
                return series.rolling(window=period, min_periods=1).mean()
        elif matype == 'WMA':
            return wma(series, period)
        else:  # default to SMA
            return series.rolling(window=period, min_periods=1).mean()

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

def compute_trend_follower_signal(df,
                                  matype='EMA',
                                  prd=20,
                                  maprd=20,
                                  rateinp=1,
                                  ulinreg=True,
                                  linprd=5):
    """
    A small wrapper function that computes the continuous Trend Follower values,
    then returns a binary signal: 1 if > 0, else 0.
    """
    trend_values = compute_trend_follower(df, matype, prd, maprd, rateinp, ulinreg, linprd)
    return np.where(trend_values > 0, 1, 0)
