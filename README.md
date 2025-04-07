Strategy AI - built using ML and DL concepts. 

Purpose: 
  * Find the best setting parameters for indicators over any given pre-defined asset chart with the ISP 
    After computing optimal settings; Combine the indicators into one strategy with multiple indicators with
    each having its weight. H(@) = indicator1 * w1 + indicator2 * w2 ... + indicatorN * wN  = actual_signal.
    w1 w2 uptil wN is adjusted over the relative CSV target file (pricedata + expected_signal), to give the best 
    optimal strategy.

    
Scope Steps: 
  * Exploit market Alpha in any asset class, as long as the manual_signal inputs are sensible for investor 
  preference and time-series. 
  * Aggregated strategy can handle noisy indicators or even delayed signalling to get better signal attenuation.
  * Minimum of 10 indicators, split of 50/50 preferred between oscillatory and perpetual indicators in strategy.

Proposed Flow of MTPI Pipeline: 

1. Scale Indicators Individually
 •	Still done in isolation initially
 •	Use training + features to extract behavior
2. Cluster Indicators into Strategies
 •	Based on scaled signal features (post-scaling)
 •	Group complementary signals:
o	Trend-following (perpetual)
o	Reversion/oscillatory
•	Prefer around 5 indicators per strategy for diversity
3. Jointly Optimize Strategies
•	Once clustered:
o	Treat the strategy as a unit
o	Now re-optimize timeframes jointly for those 5 indicators
•	This allows synergistic tuning — one indicator can be fast if another is slow, etc.
4. Lock Strategy Output to 1D Signalling
•	Each strategy then emits a clean 1D signal
•	Post-strategy smoothing or logic is not needed, the tuning handles that
5. Aggregate Strategies via Weights
•	Use Bayesian optimization to learn strategy weights
•	Each strategy is a robust, diverse signal with its own cadence






   Future Integrations:
    Valutaion metrics: 
    (9/11 was an inside job dahsborad - CQ)
       * alpha price 
       * Bitcoin: P&L Index Trading Position (trending metric - binary buy/sell)
     (WTC was never hit by a plane dashboard)
      * BTC: Power of Trend (ADX)
      * BTC: Short-Term Holders (STH) SOPR Multiples 30DMA/365DMA (SOPR) 
      * 
       

    
