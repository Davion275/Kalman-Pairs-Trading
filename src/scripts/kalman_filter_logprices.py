import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

class KalmanFilterRegression:
    def __init__(self, Q=1e-5, R=1e-4, init_state_mean=None, init_state_cov=None):
        self.n_state = 2  # State: [Beta, Alpha] (Slope, Intercept)
        
        # 1. Load previous state or initialize cold
        if init_state_mean is not None:
            self.state_mean = np.array(init_state_mean, dtype=float)
        else:
            self.state_mean = np.zeros(self.n_state)
            
        # 2. Load previous covariance or initialize cold
        if init_state_cov is not None:
            self.state_cov = np.array(init_state_cov, dtype=float)
        else:
            self.state_cov = np.zeros((self.n_state, self.n_state))
            
        self.process_noise = Q * np.eye(self.n_state)
        self.R = R 

    def update(self, x, y):
        """
        Perform one step of the Kalman Filter.
        x: Independent variable (Price of Asset B)
        y: Dependent variable (Price of Asset A)
        """
        
        # --- PREDICTION STEP ---
        # Prior State (theta_minus) = Previous State (Random Walk assumption)
        # Prior Covariance (P_minus) = Previous Covariance + Process Noise
        P_minus = self.state_cov + self.process_noise
        
        # --- OBSERVATION / SIGNAL STEP ---
        # Observation matrix H = [x, 1]  (because y = beta*x + alpha*1)
        H = np.array([x, 1.0])
        
        # Measurement Prediction: y_hat = H * state_mean
        y_hat = H @ self.state_mean
        
        # Prediction Error (Residual)
        # Asset Y - Asset X
        e = y - y_hat
        
        # Measurement Prediction Variance (S) = H * P_minus * H.T + R
        S = (H @ P_minus @ H.T) + self.R
        
        # Kalman Gain (K) = P_minus * H.T / S
        # Determines how much we trust the new data vs our model
        K = (P_minus @ H.T) / S
        
        # --- UPDATE STEP ---
        # New State = Prior State + K * Error
        self.state_mean = self.state_mean + K * e
        
        # New Covariance = (I - K * H) * P_minus
        self.state_cov = (np.eye(self.n_state) - np.outer(K, H)) @ P_minus
        
        # Return Beta (slope), Alpha (intercept), and the Spread (error)
        z_score = e / np.sqrt(S)
        beta = self.state_mean[0]

        return beta, z_score

def z_score_array(spreads_rolling):
    """
    Calculate the Z-score of the spread using a rolling window.
    spreads: are in 2d array format
    window: The number of periods to use for calculating mean and std
    """
    mean = np.mean(spreads_rolling)
    std = np.std(spreads_rolling)
    z_score = (spreads_rolling[-1] - mean) / std
    return z_score

def z_score_df(spreads_rolling):
    """
    Calculate the Z-score of the spread using a rolling window.
    spreads: Pandas Series
    window: The number of periods to use for calculating mean and std
    """
    mean = spreads_rolling.mean()
    std = spreads_rolling.std()
    z_score = (spreads_rolling - mean) / std
    return z_score


def Apply_KF_Strategy_1P(df, window=25, Q=1e-5, R=1e-4):
    # Itialization
    kf = KalmanFilterRegression(Q=Q, R=R)
    
    # Lists to store the outputs
    betas = []
    spreads = []
    z_scores = []

    # Rolling window size for Z-score calculation
    window = 25
    data = np.log(df)

    # Calculating Betas and Spreads
    for t in range(len(data)):
        # Define X (Predictor) and Y (Target)
        # We are modeling: Stock A = Beta * (Stock B) + Alpha
        x_price = data.iloc[t,0]
        y_price = data.iloc[t,1]
        
        # Run the filter update
        beta, spread_prediction = kf.update(x_price, y_price)

        # Store results
        data.at[data.index[t], 'Betas'] = beta
        data.at[data.index[t], 'Spread'] = spread_prediction

        if t >= window:
            z_score = z_score_df(data['Spread'].iloc[t-window:t])
            data.at[data.index[t], 'Z_Scores'] = z_score.iloc[-1]  # Store the latest Z-score

    # Print Object types for debugging
    print(f"Betas type: {type(betas)}, length: {len(betas)}")
    print(f"Spreads type: {type(spreads)}, length: {len(spreads)}")
    print(f"Z-scores type: {type(z_scores)}, length: {len(z_scores)}")


    data = data[~np.isnan(data['Z_Scores'])]  # Drop rows where Z-score is NaN (initial rows before we have enough data for rolling window)

    return data

def PlotsPriceBased(df):
    """
    Plotting the Stock prices, hedge ratios, and z-scores with entry and exit thresholds
    """
    data = df.copy()
    stock1 = pair[0]
    stock2 = pair[1]
    
    # Plotting 
    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Prices
    ax[0].plot(data.iloc[:,1], label=f'{stock2} (Target)')
    ax[0].plot(data.iloc[:,0], label=f'{stock1} (Predictor)')
    ax[0].set_title('Asset Prices')
    ax[0].legend()
    
    # Plot 2: Dynamic Beta
    ax[1].plot(data.index, data['Betas'], color='orange', label='Kalman Filter Beta')
    ax[1].set_title('Dynamic Hedge Ratio (Beta)')
    ax[1].legend()
    
    # Plot 3: The Spread (Trading Signal)
    ax[2].plot(data.index, data['Z_Scores'], color='green', label='Scores')
    ax[2].axhline(0, color='black', linestyle='--')
        #ax[2].set_ylim(-3,3)
    ax[2].set_title(f'Scores (Trading Signal), ({stock2} - {stock1})')
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig('trading_signal_plot.png', dpi=300, bbox_inches='tight') 

if __name__ == "__main__":
    # Example usage
    pair = ['XOM', 'CVX']
    file_path = f"datas/{pair[0]}_Au.csv"
    df = pd.read_csv(file_path)
    df = df[['date', 'close']] # Keep only date and close columns
    df = df.set_index('date') # Set the date column as index
    df.index = pd.to_datetime(df.index, format='%Y%m%d') # Convert index to datetime
    df.rename(columns={'close': pair[0]}, inplace=True)

    file_path2 = f"datas/{pair[1]}_Au.csv"
    df2 = pd.read_csv(file_path2)
    df2 = df2[['date', 'close']] # Keep only date and close columns
    df2 = df2.set_index('date') # Set the date column as index
    df2.index = pd.to_datetime(df2.index, format='%Y%m%d') # Convert index to datetime
    df2.rename(columns={'close': pair[1]}, inplace=True)

    combined_df = pd.concat([df, df2], axis=1, join='inner')
    
    result_df = Apply_KF_Strategy_1P(combined_df, window=25, Q=1e-5, R=1e-4)

    print(result_df)
    print(result_df.to_numpy())

    # PlotsPriceBased(result_df)

    