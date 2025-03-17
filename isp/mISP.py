import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_file):
    """Load CSV data and ensure proper formatting."""
    df = pd.read_csv(csv_file)
    df.columns = [col.lower() for col in df.columns]  # force lowercase
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('time', inplace=True)
    df.set_index('time', inplace=True)
    df['manual_signal'] = df['manual_signal'].ffill().fillna(0).astype(int)
    return df

def backtest_isp(df, initial_equity=10000.0):
    """
    Simulate trading based on the target ISP (manual_signal):
      - When signal==1: invest fully (apply daily return)
      - When signal==0: hold cash (equity remains unchanged)
    Returns an equity curve as a Pandas Series.
    """
    prices = df['close'].values
    # Compute daily percentage returns
    daily_returns = pd.Series(prices).pct_change().fillna(0).to_numpy()
    signals = df['manual_signal'].values
    equity = [initial_equity]
    for i in range(1, len(daily_returns)):
        if signals[i] == 1:
            equity.append(equity[-1] * (1 + daily_returns[i]))
        else:
            equity.append(equity[-1])
    return pd.Series(equity, index=df.index)

def plot_btc_with_signals(csv_file):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Forward-fill manual_signal to avoid NaNs
    df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
    
    # Create a boolean mask for when the signal changes
    df['signal_change'] = df['manual_signal'].ne(df['manual_signal'].shift(1))
    
    # Filter rows where signal changes occur
    signals_df = df[df['signal_change']].copy()
    
    # Print timestamps where signal changes occur
    timestamp = signals_df['time']
    date = pd.to_datetime(timestamp, unit='s')
    print("Timestamps (in seconds) where manual_signal changes:")
    print(signals_df[['time', 'manual_signal']], date)
    
    # Convert 'time' column to datetime and set as index for plotting
    df['DateTime'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    
    # Calculate the equity curve based on target signals
    equity_curve = backtest_isp(df, initial_equity=1.0)
    
    # Create subplots: top for price and signals, bottom for equity curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Top subplot: BTC Price with vertical lines at signal changes.
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
    transitions = np.where(df['signal_change'])[0]
    for idx in transitions:
        dt = df.index[idx]
        sig = df['manual_signal'].iloc[idx]
        if sig == 1:
            ax1.axvline(x=dt, color='green', linestyle='--', alpha=0.7,
                        label='Enter Signal' if 'Enter Signal' not in ax1.get_legend_handles_labels()[1] else "")
        elif sig == 0:
            ax1.axvline(x=dt, color='red', linestyle='--', alpha=0.7,
                        label='Exit Signal' if 'Exit Signal' not in ax1.get_legend_handles_labels()[1] else "")
    ax1.set_title("BTC Price with Manual ISP Signal Changes")
    ax1.set_ylabel("Price")
    ax1.legend()
    
    # Bottom subplot: Equity Curve from backtest.
    ax2.plot(df.index, equity_curve, label='Equity Curve', color='orange', linewidth=2)
    ax2.set_title("Backtested Equity Curve (Following Target ISP)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Equity")
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print(equity_curve[-1])

# Example usage:
if __name__ == "__main__":
    plot_btc_with_signals("./CSVdata/target.csv")
