import pandas as pd
import matplotlib.pyplot as plt

def plot_btc_with_signals(csv_file):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Filter out rows with a manual signal (non-null values)
    signals_df = df[df['manual_signal'].notna()].copy()
    
    # Print the original timestamp (in seconds) and corresponding manual signal
    print("Timestamps (in seconds) with manual signals:")
    print(signals_df[['time', 'manual_signal']])
    
    # For plotting, convert the 'time' column to datetime format
    df['DateTime'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    
    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot the close price
    ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
    
    # Loop through the filtered signals and add vertical lines at each signal candle.
    for _, row in signals_df.iterrows():
        # Convert the original time (in seconds) to datetime for plotting.
        dt = pd.to_datetime(row['time'], unit='s')
        signal = row['manual_signal']
        if signal == 1:
            ax.axvline(x=dt, color='green', linestyle='--', alpha=0.7, 
                       label='Enter Signal' if 'Enter Signal' not in ax.get_legend_handles_labels()[1] else "")
        elif signal == 0:
            ax.axvline(x=dt, color='red', linestyle='--', alpha=0.7, 
                       label='Exit Signal' if 'Exit Signal' not in ax.get_legend_handles_labels()[1] else "")
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("BTC Price with Manual Entry/Exit Signals")
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace 'btc_data.csv' with the path to your CSV file
    plot_btc_with_signals("./CSVdata/target.csv")
