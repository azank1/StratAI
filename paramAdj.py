import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# For reproducibility
torch.manual_seed(42)

# ===============================================================
# Define a module that holds the parameters for 5 DIFFERENT indicators.
#
# We'll simulate 5 indicators with different parameter schemas:
#
# Indicator 1 (Hull Suite):
#   - dimension: 5
#   - schema: [period (continuous), src (categorical: 3 options), smoothing (binary)]
#     => Here, we represent "src" by 3 numbers (expected one-hot).
#
# Indicator 2 (Q Trend):
#   - dimension: 6
#   - schema: [atrPeriod (continuous), src (categorical: 2 options),
#              signalMode (categorical: 2 options), smoothing (binary)]
#
# Indicator 3 (DEMARSI):
#   - dimension: 4, all continuous (e.g., demaPeriod, rsiPeriod, longThr, shortThr)
#
# Indicator 4 (Trend Magic):
#   - dimension: 2, both continuous (e.g., cciPeriod, atrMulti)
#
# Indicator 5 (Kalman Price Filter):
#   - dimension: 5
#   - schema: [period (continuous), gain (continuous), src (categorical: 3 options)]
# ===============================================================

class IndicatorTuner(nn.Module):
    def __init__(self):
        super(IndicatorTuner, self).__init__()
        # Initialize parameters for each indicator separately.
        # We'll initialize them to force desired initial signals.
        # We want the indicator signal = sigmoid(mean(row)).
        # For an output ~0, we can use a negative mean (e.g., -5).
        # For an output ~1, we use a positive mean (e.g., +5).
        
        # Indicator 1: target initial signal ~ 0 => initialize with -5
        self.ind1 = nn.Parameter(torch.full((5,), -5.0))  # shape (5,)
        
        # Indicator 2: target initial signal ~ 1 => initialize with +5
        self.ind2 = nn.Parameter(torch.full((6,), 5.0))   # shape (6,)
        
        # Indicator 3: target initial signal ~ 0 => initialize with -5
        self.ind3 = nn.Parameter(torch.full((4,), -5.0))  # shape (4,)
        
        # Indicator 4: target initial signal ~ 0 => initialize with -5
        self.ind4 = nn.Parameter(torch.full((2,), -5.0))  # shape (2,)
        
        # Indicator 5: target initial signal ~ 0 => initialize with -5
        self.ind5 = nn.Parameter(torch.full((5,), -5.0))  # shape (5,)
    
    def forward(self):
        # For each indicator, compute the signal = sigmoid(mean(parameters))
        s1 = torch.sigmoid(self.ind1.mean())
        s2 = torch.sigmoid(self.ind2.mean())
        s3 = torch.sigmoid(self.ind3.mean())
        s4 = torch.sigmoid(self.ind4.mean())
        s5 = torch.sigmoid(self.ind5.mean())
        
        # Return a tensor of individual signals and the aggregated signal (average)
        signals = torch.stack([s1, s2, s3, s4, s5])  # shape: [5]
        aggregator = signals.mean()
        return signals, aggregator

# ===============================================================
# Instantiate the model
model = IndicatorTuner()

# ===============================================================
# Check initial signals and aggregated output
initial_signals, initial_agg = model.forward()
print("Initial Indicator Signals:")
print(initial_signals.detach().numpy())
print("Initial Aggregated Signal (should be ~0.2):", initial_agg.item())

# ===============================================================
# Our training goal: We want the aggregated signal to be 1 (positive trend).
target = torch.tensor([1.0])

# Use Adam optimizer on all indicator parameters
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

num_epochs = 200

print("\nTraining for 200 epochs...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    _, agg = model.forward()
    loss = criterion(agg.view(1), target)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.4f} | Aggregator: {agg.item():.4f}")

final_signals, final_agg = model.forward()
print("\nFinal Aggregated Signal:", final_agg.item())
print("Final Raw Indicator Parameters:")
print("Indicator 1:", model.ind1.data.numpy())
print("Indicator 2:", model.ind2.data.numpy())
print("Indicator 3:", model.ind3.data.numpy())
print("Indicator 4:", model.ind4.data.numpy())
print("Indicator 5:", model.ind5.data.numpy())

# ===============================================================
# Post-Processing: Convert Raw Parameters to Valid Settings
# ---------------------------------------------------------------
# For each indicator, we know the schema.
# We'll post-process as follows:
#
# Indicator 1 (Hull Suite):
#   - [0]: period (continuous) -> clamp to [1, 100]
#   - [1:4]: one-hot for source (3 options: "high", "low", "close") -> pick argmax, set that to 1, others 0
#   - [4]: smoothing (binary) -> threshold at 0.5
#
# Indicator 2 (Q Trend):
#   - [0]: atrPeriod (continuous) -> clamp to [1, 100]
#   - [1:3]: one-hot for source (2 options) -> since there are 2, use indices [1,2] (we'll ignore index3 if any)
#         Actually, here we assume positions [1,2] for source (2 options) and [3,4] for signalMode (2 options)
#         and index [5] for smoothing.
#   - For simplicity, we'll do:
#         * Clamp index0 to [1,100]
#         * For indices [1,2]: argmax => one-hot vector of length 2.
#         * For indices [3,4]: argmax => one-hot vector of length 2.
#         * For index 5: threshold at 0.5.
#
# Indicator 3 (DEMARSI): All continuous. We'll clamp each to [1, 100].
#
# Indicator 4 (Trend Magic): Two continuous. Clamp both to [1, 100].
#
# Indicator 5 (Kalman Price Filter):
#   - [0]: period (continuous) -> clamp to [1, 100]
#   - [1:4]: one-hot for source (3 options) -> argmax over indices 1,2,3.
#   - [4]: gain (continuous) -> clamp to [1,50].

def post_process_indicator(params_tensor, schema):
    """
    params_tensor: a 1D tensor of raw parameters for one indicator.
    schema: a dictionary defining the schema for the indicator.
    
    The schema should contain:
      - 'continuous': list of indices with (min, max) constraints.
      - 'categorical': list of tuples (start_index, length, options)
      - 'binary': list of indices (or a list of tuples with threshold)
    
    This function returns a processed tensor.
    """
    proc = params_tensor.clone()
    # Process continuous parameters
    for idx, (mn, mx) in schema.get('continuous', {}).items():
        proc[idx] = torch.clamp(proc[idx], min=mn, max=mx)
    
    # Process categorical parameters
    # For each categorical, get start index and length, then set that block to one-hot
    for (start, length, options) in schema.get('categorical', []):
        # We take the argmax in the block
        block = proc[start : start+length]
        argmax_val = torch.argmax(block).item()
        proc[start : start+length] = 0
        proc[start + argmax_val] = 1.0
    
    # Process binary parameters
    for idx in schema.get('binary', []):
        proc[idx] = 1.0 if proc[idx] >= 0.5 else 0.0
        
    return proc

# Define schema for each indicator:

schema1 = {
    'continuous': {0: (1.0, 100.0)},
    'categorical': [(1, 3, ["high", "low", "close"])],
    'binary': [4],
}

schema2 = {
    'continuous': {0: (1.0, 100.0)},
    'categorical': [(1, 2, ["high", "close"]), (3, 2, ["TypeA", "TypeB"])],
    'binary': [5],
}

schema3 = {
    'continuous': {0: (1.0, 100.0), 1: (1.0, 100.0), 2: (1.0, 100.0), 3: (1.0, 100.0)},
    # No categorical or binary for indicator 3.
}

schema4 = {
    'continuous': {0: (1.0, 100.0), 1: (1.0, 100.0)},
}

schema5 = {
    'continuous': {0: (1.0, 100.0), 4: (1.0, 50.0)},
    'categorical': [(1, 3, ["high", "low", "close"])],
    # No binary in indicator 5.
}

# Post-process each indicator's parameters
final_ind1 = post_process_indicator(model.ind1.data, schema1)
final_ind2 = post_process_indicator(model.ind2.data, schema2)
final_ind3 = post_process_indicator(model.ind3.data, schema3)
final_ind4 = post_process_indicator(model.ind4.data, schema4)
final_ind5 = post_process_indicator(model.ind5.data, schema5)

print("\nPost-Processed Final Parameters:")

print("Indicator 1:", final_ind1.numpy())
print("Indicator 2:", final_ind2.numpy())
print("Indicator 3:", final_ind3.numpy())
print("Indicator 4:", final_ind4.numpy())
print("Indicator 5:", final_ind5.numpy())

# Translate these into high-level settings
def interpret_indicator(ind_tensor, schema, indicator_name):
    settings = {}
    if indicator_name == "Indicator 1":
        settings['period'] = ind_tensor[0].item()
        # Decode categorical: indices 1,2,3
        cat_val = torch.argmax(ind_tensor[1:4]).item()
        source_map = {0: "high", 1: "low", 2: "close"}
        settings['source'] = source_map[cat_val]
        settings['smoothing'] = int(ind_tensor[4].item())
    elif indicator_name == "Indicator 2":
        settings['atrPeriod'] = ind_tensor[0].item()
        # For source: indices 1-2
        cat_val1 = torch.argmax(ind_tensor[1:3]).item()
        source_map2 = {0: "high", 1: "close"}
        settings['source'] = source_map2[cat_val1]
        # For signalMode: indices 3-4
        cat_val2 = torch.argmax(ind_tensor[3:5]).item()
        mode_map = {0: "TypeA", 1: "TypeB"}
        settings['signalMode'] = mode_map[cat_val2]
        settings['smoothing'] = int(ind_tensor[5].item())
    elif indicator_name == "Indicator 3":
        settings['demaPeriod'] = ind_tensor[0].item()
        settings['rsiPeriod'] = ind_tensor[1].item()
        settings['longThreshold'] = ind_tensor[2].item()
        settings['shortThreshold'] = ind_tensor[3].item()
    elif indicator_name == "Indicator 4":
        settings['cciPeriod'] = ind_tensor[0].item()
        settings['atrMulti'] = ind_tensor[1].item()
    elif indicator_name == "Indicator 5":
        settings['period'] = ind_tensor[0].item()
        # For source: indices 1-3
        cat_val3 = torch.argmax(ind_tensor[1:4]).item()
        source_map5 = {0: "high", 1: "low", 2: "close"}
        settings['source'] = source_map5[cat_val3]
        settings['gain'] = ind_tensor[4].item()
    return settings

print("\nInterpreted Settings:")
print("Indicator 1:", interpret_indicator(final_ind1, schema1, "Indicator 1"))
print("Indicator 2:", interpret_indicator(final_ind2, schema2, "Indicator 2"))
print("Indicator 3:", interpret_indicator(final_ind3, schema3, "Indicator 3"))
print("Indicator 4:", interpret_indicator(final_ind4, schema4, "Indicator 4"))
print("Indicator 5:", interpret_indicator(final_ind5, schema5, "Indicator 5"))
