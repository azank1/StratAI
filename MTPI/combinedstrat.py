import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# 1) Import indicator functions from your 'indicators' folder
#    (Make sure the files and function names match your actual code)
from indicators.HullSindicator import hull_suite_indicator
from indicators.TrendFollower import compute_trend_follower_signal

###########################################
# Utility Functions
###########################################

def load_json_settings(filename):
    """
    Load a JSON file (with optimal indicator settings) from the 'settings' folder.
    """
    path = os.path.join("MTPI", filename)
    with open(path, "r") as f:
        return json.load(f)

def aggregate_signals(w1, w2, signalA, signalB):
    """
    Weighted average of two indicator signals (signalA, signalB),
    thresholded at 0.5 to produce a final binary signal.
    final_signal = 1 if (w1*signalA + w2*signalB)/(w1+w2) > 0.5, else 0.
    """
    total_weight = w1 + w2
    combined = (w1 * signalA + w2 * signalB) / total_weight
    final_signal = np.where(combined > 0.5, 1, 0)
    return final_signal

###########################################
# Main Combined Strategy
###########################################
def main():
    # 2) Load BTC CSV data
    df = pd.read_csv("./CSVdata/target.csv")
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('time', inplace=True)
    df.set_index('time', inplace=True)
    
    # If there's a 'manual_signal' in your CSV, forward-fill if needed
    if 'manual_signal' in df.columns:
        df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
        target = df['manual_signal'].values
    else:
        # If no manual signal is present, we won't be able to optimize to match it
        target = np.zeros(len(df))
    
    close_prices = df['close'].values
    
    # 3) Load each indicator's saved optimal settings from 'settings/' folder
    hull_settings = load_json_settings("./settings/hull_suite_settings.json")
    trend_settings = load_json_settings("./settings/trend_follower_settings.json")
    
    # 4) Compute each indicator's binary signal using the pre-defined functions
    #    from your 'indicators' folder
    hull_signal = hull_suite_indicator(
        df,
        src_col=hull_settings["source"],
        mode=hull_settings["mode"],
        length=hull_settings["length"],
        length_mult=hull_settings["length_mult"]
    )
    
    trend_signal = compute_trend_follower_signal(
    df,
    matype=trend_settings["matype"],
    prd=trend_settings["prd"],
    maprd=trend_settings["maprd"],
    rateinp=trend_settings["rateinp"],
    ulinreg=trend_settings["ulinreg"],
    linprd=trend_settings["linprd"]
)
    
    hull_signal = np.array(hull_signal)
    trend_signal = np.array(trend_signal)
    
    # 5) Define an objective function to optimize the weights w1, w2
    def objective(w1, w2):
        final_signal = aggregate_signals(hull_signal, trend_signal, w1, w2)
        mae = np.mean(np.abs(final_signal - target))
        return -mae  # negative for BayesianOptimization (maximizes)
    
    pbounds = {
        'w1': (0.1, 10),
        'w2': (0.1, 10)
    }
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    # 6) Optimize weights
    optimizer.maximize(init_points=5, n_iter=20)
    
    best_params = optimizer.max['params']
    best_loss = -optimizer.max['target']
    w1_opt = best_params['w1']
    w2_opt = best_params['w2']
    
    print("Optimal Weights:")
    print(f"w1 = {w1_opt:.4f}, w2 = {w2_opt:.4f}")
    print(f"Best MAE: {best_loss:.4f}")
    
    # # 7) Compute final aggregated signal & equity curve
    # final_signal = aggregate_signals(hull_signal, trend_signal, w1_opt, w2_opt)
    # equity_curve = compute_equity_curve(close_prices, final_signal)
    
    # # 8) Plot results
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # # Top subplot: Price with indicator signals
    # ax1.plot(df.index, close_prices, label='BTC Price', color='black', linewidth=1.5)
    # ax1.plot(df.index, hull_signal, label='Hull Signal', color='blue', linestyle='--', alpha=0.7)
    # ax1.plot(df.index, trend_signal, label='Trend Signal', color='red', linestyle='--', alpha=0.7)
    # ax1.set_title("BTC Price with Indicator Signals (Hull + Trend)")
    # ax1.set_ylabel("Price / Signals")
    # ax1.legend(loc='upper left')
    
    # # Bottom subplot: Equity
    # ax2.plot(df.index, equity_curve, label='Equity Curve', color='orange', linewidth=2)
    # ax2.set_title("Equity Curve from Aggregated Strategy")
    # ax2.set_xlabel("Date")
    # ax2.set_ylabel("Equity")
    # ax2.legend(loc='upper left')
    
    # plt.tight_layout()
    # plt.show()
    

if __name__ == "__main__":
    main()