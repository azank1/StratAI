import pandas as pd
import matplotlib.pyplot as plt

def plot_btc_with_signals(csv_file):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Forward-fill manual_signal if needed (to avoid NaNs)
    df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
    
    # Create a boolean mask where the signal is different from the previous row
    df['signal_change'] = df['manual_signal'].ne(df['manual_signal'].shift(1))
    
    # Filter to only the rows where a change actually occurs
    signals_df = df[df['signal_change']].copy()
    
    # Print the original timestamps and signals only where a change occurs
    print("Timestamps (in seconds) where manual_signal changes:")
    print(signals_df[['time', 'manual_signal']])
    
    # Convert 'time' column to datetime for plotting
    df['DateTime'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    
    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot the close price
    ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
    
    # For each row where the signal changes, draw a vertical line
    for _, row in signals_df.iterrows():
        # Convert the original time (in seconds) to datetime for plotting
        dt = pd.to_datetime(row['time'], unit='s')
        signal = row['manual_signal']
        
        if signal == 1:
            ax.axvline(
                x=dt, color='green', linestyle='--', alpha=0.7,
                label='Enter Signal' if 'Enter Signal' not in ax.get_legend_handles_labels()[1] else ""
            )
        elif signal == 0:
            ax.axvline(
                x=dt, color='red', linestyle='--', alpha=0.7,
                label='Exit Signal' if 'Exit Signal' not in ax.get_legend_handles_labels()[1] else ""
            )
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("BTC Price with Manual Signals (Only Where Signal Changes)")
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    plot_btc_with_signals("./CSVdata/updated_target.csv")
