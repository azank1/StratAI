import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wma(series, window):
    """
    Calculate the Weighted Moving Average (WMA) of a series over a given window.
    """
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def HMA(src, length):
    """
    Calculate the Hull Moving Average (HMA) for the given source series and length.
    Formula: HMA = WMA(2*WMA(src, length/2) - WMA(src, length), round(sqrt(length)))
    """
    half_length = int(round(length / 2))
    sqrt_length = int(round(np.sqrt(length)))
    wma_half = wma(src, half_length)
    wma_full = wma(src, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

def EHMA(src, length):
    """
    Calculate the Exponential Hull Moving Average (EHMA) for the given source series and length.
    Formula: EHMA = EMA(2*EMA(src, length/2) - EMA(src, length), round(sqrt(length)))
    """
    half_length = int(round(length / 2))
    sqrt_length = int(round(np.sqrt(length)))
    ema_half = src.ewm(span=half_length, adjust=False).mean()
    ema_full = src.ewm(span=length, adjust=False).mean()
    diff = 2 * ema_half - ema_full
    return diff.ewm(span=sqrt_length, adjust=False).mean()

def THMA(src, length):
    """
    Calculate the Triple Hull Moving Average (THMA) for the given source series and length.
    Formula: THMA = WMA(WMA(src, length/3)*3 - WMA(src, length/2) - WMA(src, length), length)
    """
    length = int(round(length))
    third_length = int(round(length / 3))
    half_length = int(round(length / 2))
    wma_third = wma(src, third_length)
    wma_half = wma(src, half_length)
    wma_full = wma(src, length)
    diff = 3 * wma_third - wma_half - wma_full
    return wma(diff, length)

def hull_suite_indicator(df, 
                         src_col="close", 
                         mode="Hma",        # Options: "Hma", "Ehma", "Thma"
                         length=55, 
                         length_mult=1.0, 
                         switch_color=True, 
                         candle_col=False,  # (placeholder) for future candle coloring.
                         visual_switch=True, 
                         thicknes_switch=1, 
                         transp_switch=40):
    """
    Calculate the Hull Suite indicator on the DataFrame `df`.
    
    Returns the DataFrame with additional columns:
      - "MHULL": The current Hull value.
      - "SHULL": The Hull value shifted by 2 periods.
      - "hull_color": a color string ("green" or "red") based on trend.
    """
    eff_length = int(round(length * length_mult))
    src = df[src_col]
    
    if mode == "Hma":
        hull = HMA(src, eff_length)
    elif mode == "Ehma":
        hull = EHMA(src, eff_length)
    elif mode == "Thma":
        eff_length_thma = eff_length / 2
        hull = THMA(src, eff_length_thma)
    else:
        raise ValueError("Invalid mode. Choose from 'Hma', 'Ehma', or 'Thma'.")
    
    hull_shifted = hull.shift(2)
    
    if switch_color:
        hull_color = np.where(hull > hull_shifted, "green", "red")
    else:
        hull_color = np.array(["orange"] * len(hull))
    
    df = df.copy()
    df["MHULL"] = hull
    df["SHULL"] = hull_shifted
    df["hull_color"] = hull_color
    return df

# ----------------------------
# Example usage:
# ----------------------------
if __name__ == "__main__":
    # Load your CSV data. Adjust file path and date column name as needed.
    # For example:
    # df = pd.read_csv("your_data.csv", parse_dates=["Date"], index_col="Date")
    
    # For demonstration purposes, we'll create a dummy DataFrame.
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    np.random.seed(42)
    prices = np.random.lognormal(mean=0, sigma=0.02, size=200).cumprod() * 100
    df = pd.read_csv("./CSVdata/BTC.csv")
    
    # Calculate the Hull Suite indicator.
    df = hull_suite_indicator(df, mode="Hma", length=55)
    
    # Plot the close price and Hull indicator.
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the price data.
    ax.plot(df.index, df["close"], label="Price", color="black", linewidth=1.5)
    
    # Plot the MHULL indicator.
    ax.plot(df.index, df["MHULL"], label="MHULL", color="blue", linewidth=2)
    
    # Plot the shifted Hull indicator (SHULL) if visual_switch is True.
    ax.plot(df.index, df["SHULL"], label="SHULL", color="gray", linewidth=2)
    
    # Fill the area between MHULL and SHULL using the hull_color array.
    # For each continuous segment with the same color, we fill separately.
    current_color = None
    start_idx = 0
    for i in range(len(df)):
        col = df["hull_color"].iloc[i]
        if current_color is None:
            current_color = col
        if col != current_color or i == len(df)-1:
            # Determine the segment range.
            end_idx = i if col != current_color else i+1
            ax.fill_between(df.index[start_idx:end_idx],
                            df["MHULL"].iloc[start_idx:end_idx],
                            df["SHULL"].iloc[start_idx:end_idx],
                            color=current_color, alpha=0.3)
            start_idx = i
            current_color = col

    ax.set_title("Price with Hull Suite Indicator")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price / Hull Value")
    ax.legend()
    plt.tight_layout()
    plt.show()
