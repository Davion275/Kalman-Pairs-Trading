import numpy as np
from collections import deque

class CumulativeReturnsKF:
    def __init__(self, Q=1e-5, R=1e-4, window=20):
        # --- KALMAN PARAMETERS ---
        self.n_state = 2  # State: [Beta, Alpha]
        self.state_mean = np.zeros(self.n_state)
        self.state_mean[0] = 1.0  # Initial Beta assumption, why???
        self.state_cov = np.ones((self.n_state, self.n_state)) # why??
        self.process_noise = Q * np.eye(self.n_state)
        self.R = R
        
        # --- STRATEGY PARAMETERS ---
        self.prev_x = None
        self.prev_y = None
        
        # To calculate Cumulative Returns starting from 0%
        self.initial_x = None
        self.initial_y = None
        
        self.spread_window = deque(maxlen=window)

    def update(self, price_x, price_y):
        """
        Input: LOG PRICES of Asset X and Asset Y
        Output: Beta, Z-Score, CumRet_Y, CumRet_X, Weighted_CumRet_X
        """
        
        # 1. Initialization (Warm Start & Baseline)
        if self.prev_x is None:
            self.prev_x = price_x
            self.prev_y = price_y
            self.initial_x = price_x
            self.initial_y = price_y
            # Return zeros for the first step as we have no change yet
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # 2. Calculate Daily Returns (For the Kalman Filter)
        ret_x = price_x - self.prev_x
        ret_y = price_y - self.prev_y
        
        # Update previous for next step
        self.prev_x = price_x
        self.prev_y = price_y

        # --- KALMAN FILTER STEP (Returns Space) ---
        P_minus = self.state_cov + self.process_noise
        
        # Observation: ret_y = Beta * ret_x + Alpha
        H = np.array([ret_x, 1.0]) 
        
        y_hat = H @ self.state_mean
        e = ret_y - y_hat
        S = (H @ P_minus @ H.T) + self.R
        K = (P_minus @ H.T) / S
        
        self.state_mean = self.state_mean + K * e
        self.state_cov = (np.eye(self.n_state) - np.outer(K, H)) @ P_minus
        
        beta_returns = self.state_mean[0]
        
        # --- CUMULATIVE RETURN CALCULATION ---
        # Calculate total cumulative return since start (0.05 = 5%)
        # Since inputs are Log Prices, CumRet is just Current - Initial
        cum_ret_y = price_y - self.initial_y
        cum_ret_x = price_x - self.initial_x
        
        # This is the "Hedge Component": How much X has moved, scaled by Beta
        weighted_cum_ret_x = beta_returns * cum_ret_x

        # --- SPREAD & SIGNAL (Price Space) ---
        # The spread is the difference between the actual Y return and the weighted X return
        # Asset Y - Asset X
        current_spread = cum_ret_y - weighted_cum_ret_x
        
        self.spread_window.append(current_spread)
        
        if len(self.spread_window) < 5:
            return beta_returns, 0.0, cum_ret_y, cum_ret_x, weighted_cum_ret_x
            
        rolling_mean = np.mean(self.spread_window)
        rolling_std = np.std(self.spread_window)
        
        if rolling_std < 1e-8:
            z_score = 0.0
        else:
            z_score = (current_spread - rolling_mean) / rolling_std
            
        return beta_returns, z_score, cum_ret_y, cum_ret_x, weighted_cum_ret_x

