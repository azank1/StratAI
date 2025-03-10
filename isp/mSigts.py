import pandas as pd

def print_signal_timestamps(csv_file, time_col="time", signal_col="manual_signal"):
    """
    Reads the CSV file and prints only the timestamps (in seconds) for candles
    that have a manual signal (non-null values in the signal column).
    
    Parameters:
      csv_file   : Path to the input CSV file.
      time_col   : Name of the column containing the timestamp (in seconds).
      signal_col : Name of the column containing the manual signal.
    """
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Filter rows that have a manual signal (non-null)
    signals_df = df[df[signal_col].notna()]
    
    # Print only the timestamp column
    print("Timestamps (in seconds) with manual signals:")
    print(signals_df[time_col])

# Example usage:
if __name__ == "__main__":
    # Replace 'btc_data.csv' with the path to your CSV file.
    print_signal_timestamps("./CSVdata/target.csv")
