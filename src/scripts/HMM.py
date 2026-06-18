import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import warnings

def generate_hmm_position_sizes(spread, base_notional=10000, window=126, high_vol_weight=0.0, low_vol_weight=1.0, high_vol_thresh=3.0, low_vol_thresh=4.0):
    # Convert to raw NumPy to strip Pandas overhead and prevent index misalignment
    spread_diff_vals = spread.diff().dropna().values 
    n_samples = len(spread_diff_vals)
    
    sizes_arr = np.zeros(n_samples)
    thresh_arr = np.full(n_samples, 4.0) 
    step = 5 
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    for i in range(window, n_samples, step):
        train_window = spread_diff_vals[i-window:i].reshape(-1, 1)
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        
        try:
            model.fit(train_window)
            var_state_0 = model.covars_[0][0][0]
            var_state_1 = model.covars_[1][0][0]
            high_vol_state_id = 0 if var_state_0 > var_state_1 else 1
            
            predictions = model.predict(train_window)
            last_state = predictions[-1]

            trade_weight = high_vol_weight if last_state == high_vol_state_id else low_vol_weight
            thresh_val = high_vol_thresh if last_state == high_vol_state_id else low_vol_thresh

            end_idx = min(i + step, n_samples)
            sizes_arr[i:end_idx] = base_notional * trade_weight
            thresh_arr[i:end_idx] = thresh_val
                
        except Exception:
            end_idx = min(i + step, n_samples)
            sizes_arr[i:end_idx] = 0.0
            thresh_arr[i:end_idx] = 4.0

    # Rebuild the Pandas Series exactly once at the end
    pad_len = len(spread) - n_samples
    final_sizes = np.concatenate([np.zeros(pad_len), sizes_arr])
    final_thresh = np.concatenate([np.full(pad_len, 4.0), thresh_arr])
    
    return pd.Series(final_sizes, index=spread.index).fillna(0.0), pd.Series(final_thresh, index=spread.index).fillna(4.0)