import pandas as pd

def add_manual_signals_to_target(target_csv, signals_csv, output_csv):
    """
    Reads the target CSV (which contains fixed timestamps and price data)
    and a separate CSV containing manual signal annotations (with columns 'timestamp' and 'signal'),
    then merges the manual signals into the target CSV by performing an outer join on the timestamp.
    
    This ensures that any new timestamps in the signals CSV are added.
    After merging, the 'manual_signal' column is forward-filled so that every row has a valid signal.
    
    Parameters:
      target_csv : str - path to the target CSV file (e.g., BTC price data)
      signals_csv: str - path to the CSV file containing manual signals (with columns 'timestamp' and 'signal')
      output_csv : str - path to write the updated CSV file
      
    Returns:
      df_merged : pandas DataFrame with the manual_signal column updated.
    """
    # Load the target CSV; assume the timestamp column is named 'time' and parse it as datetime.
    df_target = pd.read_csv(target_csv, parse_dates=['time'])
    
    # Load the manual signals CSV; assume columns: 'timestamp' and 'signal'
    df_signals = pd.read_csv(signals_csv, parse_dates=['timestamp'])
    
    # Rename the columns in the signals DataFrame to match the target
    df_signals.rename(columns={'timestamp': 'time', 'signal': 'manual_signal'}, inplace=True)
    
    # Merge the two DataFrames on the 'time' column using an outer join to include all timestamps
    df_merged = pd.merge(df_target, df_signals[['time', 'manual_signal']], on='time', how='outer')
    
    # Sort by time to maintain chronological order
    df_merged.sort_values('time', inplace=True)
    
    # If the target CSV already had a manual_signal column, it will appear as manual_signal_x.
    # The new signals come in as manual_signal_y.
    if 'manual_signal_x' in df_merged.columns and 'manual_signal_y' in df_merged.columns:
        # Give preference to the new signals from signals CSV
        df_merged['manual_signal'] = df_merged['manual_signal_y'].combine_first(df_merged['manual_signal_x'])
        df_merged.drop(columns=['manual_signal_x', 'manual_signal_y'], inplace=True)
    # Otherwise, the merged column is already named manual_signal.
    
    # Forward-fill any missing manual_signal values (or fill with 0 if preferred)
    df_merged['manual_signal'] = df_merged['manual_signal'].ffill().fillna(0)
    
    # Save the updated DataFrame to the specified output CSV.
    df_merged.to_csv(output_csv, index=False)
    
    return df_merged

# Example usage:
if __name__ == "__main__":
    updated_df = add_manual_signals_to_target("./CSVdata/target.csv", "./CSVdata/signals.csv", "./CSVdata/updated_target.csv")
    print(updated_df.head())
