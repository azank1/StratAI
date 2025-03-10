Strategy AI - built using ML and DL concepts. 

Purpose: 
  * Find the best setting parameters for indicators over any given pre-defined asset chart with the ISP 
    After computing optimal settings; Combine the indicators into one strategy with multiple indicators with
    each having its weight. H(@) = indicator1 * w1 + indicator2 * w2 ... + indicatorN * wN  = actual_signal.
    w1 w2 uptil wN is adjusted over the relative CSV target file (pricedata + expected_signal), to give the best 
    optimal strategy.

    
Scope: 
  * Exploit market Alpha in any asset class, as long as the manual_signal inputs are sensible for investor 
  preference and time-series. 
  * Aggregated strategy can handle noisy indicators or even delayed signalling to get better signal attenuation.
  * Minimum of 10 indicators, split of 50/50 preferred between oscillatory and perpetual indicators in strategy.


   Future Integrations:
    * Valuation aggregated into overall trend. 
    * Strategic DCA based on valuation 
    * fundamental analysis of macro data
    * senitment analysis (sentix/fearngreed) 
    * Longer Term Technical analysis.
    
