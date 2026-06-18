import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.generic.splitters import RangeSplitter
import plotly.io as plo
from datetime import datetime
import os
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit
import concurrent.futures
from tqdm import tqdm
import gc

from HMM import generate_hmm_position_sizes

# --- 1. PURE NUMBA MATH ENGINES ---
@njit(cache=True)
def fast_kalman_filter(x_prices, y_prices, Q, R, init_beta, init_alpha, init_c00, init_c01, init_c10, init_c11):
    n = len(x_prices)
    beta_array = np.empty(n, dtype=np.float64)
    z_scores_array = np.empty(n, dtype=np.float64)
    s_beta = np.empty(n, dtype=np.float64)
    s_alpha = np.empty(n, dtype=np.float64)
    c00_arr = np.empty(n, dtype=np.float64)
    c01_arr = np.empty(n, dtype=np.float64)
    c10_arr = np.empty(n, dtype=np.float64)
    c11_arr = np.empty(n, dtype=np.float64)
    
    state_mean = np.array([init_beta, init_alpha], dtype=np.float64)
    state_cov = np.array([[init_c00, init_c01], [init_c10, init_c11]], dtype=np.float64)
    process_noise = np.array([[Q, 0.0], [0.0, Q]], dtype=np.float64)
    
    for t in range(n):
        x, y = x_prices[t], y_prices[t]
        P_minus = state_cov + process_noise
        H_0, H_1 = x, 1.0
        y_hat = H_0 * state_mean[0] + H_1 * state_mean[1]
        e = y - y_hat
        
        S = (H_0 * P_minus[0, 0] * H_0 + H_0 * P_minus[0, 1] * H_1 + 
             H_1 * P_minus[1, 0] * H_0 + H_1 * P_minus[1, 1] * H_1) + R
             
        K0 = (P_minus[0, 0] * H_0 + P_minus[0, 1] * H_1) / S
        K1 = (P_minus[1, 0] * H_0 + P_minus[1, 1] * H_1) / S
        
        state_mean[0] = state_mean[0] + K0 * e
        state_mean[1] = state_mean[1] + K1 * e
        
        state_cov[0, 0] = (1 - K0 * H_0) * P_minus[0, 0] - (K0 * H_1) * P_minus[1, 0]
        state_cov[0, 1] = (1 - K0 * H_0) * P_minus[0, 1] - (K0 * H_1) * P_minus[1, 1]
        state_cov[1, 0] = -(K1 * H_0) * P_minus[0, 0] + (1 - K1 * H_1) * P_minus[1, 0]
        state_cov[1, 1] = -(K1 * H_0) * P_minus[0, 1] + (1 - K1 * H_1) * P_minus[1, 1]
        
        beta_array[t] = state_mean[0]
        z_scores_array[t] = e / np.sqrt(S)
        s_beta[t], s_alpha[t] = state_mean[0], state_mean[1]
        c00_arr[t], c01_arr[t], c10_arr[t], c11_arr[t] = state_cov[0, 0], state_cov[0, 1], state_cov[1, 0], state_cov[1, 1]
        
    return beta_array, z_scores_array, s_beta, s_alpha, c00_arr, c01_arr, c10_arr, c11_arr

# 1. Remove the @njit decorator and add *args, **kwargs to absorb VectorBT's background inputs
# --- REPLACE: Your apply_signal_logic_fast function ---
def apply_signal_logic_fast(z_scores, dynamic_blowout_thresh, entry_thresh, exit_thresh, *args, **kwargs):
    # Convert to numpy array for efficient vectorization
    z = np.asarray(z_scores)
    dt = np.asarray(dynamic_blowout_thresh)
    
    # Create a 1-day shifted array to detect exact crossovers
    z_shifted = np.roll(z, 1, axis=0)
    z_shifted[0] = z[0] 
    
    # Use dt array instead of a static float
    entries = (z < -entry_thresh) & (z_shifted >= -entry_thresh) & (np.abs(z) < dt)
    short_entries = (z > entry_thresh) & (z_shifted <= entry_thresh) & (np.abs(z) < dt)
    
    normal_exits = (z >= -exit_thresh) & (z_shifted < -exit_thresh)   
    normal_short_exits = (z <= exit_thresh) & (z_shifted > exit_thresh)
    
    # Blowout exits now evaluate against the dynamic array
    blowout_exits = (z <= -dt) & (z_shifted > -dt)
    blowout_short_exits = (z >= dt) & (z_shifted < dt)
    
    final_exits = normal_exits | blowout_exits
    final_short_exits = normal_short_exits | blowout_short_exits
    
    return entries, final_exits, short_entries, final_short_exits

# 2. Add **kwargs to the Kalman Filter adapter as well, just to make it completely bulletproof
def Apply_KF_Strategy_1P_optimized(x_prices, y_prices, Q=1e-5, R=1e-4, 
                                   init_beta=0.0, init_alpha=0.0, 
                                   init_c00=0.0, init_c01=0.0, init_c10=0.0, init_c11=0.0, **kwargs):
    
    # Flatten VectorBT's 2D matrices into 1D arrays
    x_log = np.log(np.asarray(x_prices, dtype=np.float64).ravel())
    y_log = np.log(np.asarray(y_prices, dtype=np.float64).ravel())
    
    return fast_kalman_filter(x_log, y_log, Q, R, init_beta, init_alpha, init_c00, init_c01, init_c10, init_c11)


# --- 2. VECTORBT FACTORIES ---
KF_Indicator = vbt.IndicatorFactory(
    input_names=['x_price', 'y_price'],
    param_names=['Q', 'R', 'init_beta', 'init_alpha', 'init_c00', 'init_c01', 'init_c10', 'init_c11'],
    output_names=['Betas', 'Z_Scores', 's_beta', 's_alpha', 'c00', 'c01', 'c10', 'c11']
).from_apply_func(Apply_KF_Strategy_1P_optimized, init_beta=0.0, init_alpha=0.0, init_c00=0.0, init_c01=0.0, init_c10=0.0, init_c11=0.0)

Signal_Indicator = vbt.IndicatorFactory(
    input_names=['z_scores', 'dynamic_blowout_thresh'],
    param_names=['entry_thresh', 'exit_thresh'],
    output_names=['entries', 'exits', 'short_entries', 'short_exits']
).from_apply_func(apply_signal_logic_fast)

# --- 3. HELPER FUNCTIONS ---
# --- REPLACE: Your load_data_for_pairs function ---
def load_data_for_pairs(pairs_list):
    dataframes = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datas_dir = os.path.join(base_dir, 'datas')
    for pair in pairs_list:
        for ticker in pair:
            if ticker not in dataframes: 
                file_path = os.path.join(datas_dir, f"{ticker}_Au.csv")
                df = pd.read_csv(file_path)
                df = df[['date', 'close']].set_index('date')
                df.index = pd.to_datetime(df.index, format='%Y%m%d')
                df.rename(columns={'close': ticker}, inplace=True)
                dataframes[ticker] = df
    # 1. Outer Join: Create the union of all dates
    master_df = pd.concat(dataframes.values(), axis=1, join='outer')
    
    # 2. Remove Duplicate Indices: Keep the first occurrence of any timestamp
    master_df = master_df[~master_df.index.duplicated(keep='first')]
    
    # 3. Sort Chronologically: Outer joins don't always preserve order
    master_df = master_df.sort_index()
    
    # 4. Filter for Business Days: Removes weekends that might slip in
    # This keeps the index clean for VectorBT's frequency checks
    master_df = master_df[master_df.index.dayofweek < 5]
    
    # 5. Handle Gaps: Forward fill prices, but drop rows that are entirely NaN
    # (e.g., a holiday where no stocks in your list traded)
    master_df = master_df.ffill(limit=5).dropna(how='all')
    
    return master_df

def check_correlation(price_x, price_y, min_corr=0.6):
    returns_x, returns_y = price_x.pct_change().dropna(), price_y.pct_change().dropna()
    corr = returns_x.corr(returns_y)
    return corr >= min_corr, corr

def test_coint_prices(price_x, price_y, p_value_thresh=0.05):
    score, pvalue, _ = coint(price_x, price_y)
    return pvalue < p_value_thresh, pvalue

# --- 4. MULTIPROCESSING WORKER FUNCTION ---
def process_single_pair(pair, splits, master_prices, master_benchmark_returns, entry_test_values, exit_test_values, allocation_per_pair, fixed_q, fixed_r, leverage=1, slippage=0.0):
    stock_x, stock_y = pair
    pair_name = f"{stock_x}-{stock_y}" 
    # print(f"[{pair_name}] Process started on core...")

    oos_prices_list, oos_long_entries_list, oos_short_entries_list = [], [], []
    oos_long_exits_list, oos_short_exits_list, oos_sizes_list, oos_z_scores_list = [], [], [], []

    trade_this_cycle_overall = False

    for split_idx, (train_bounds, test_bounds) in enumerate(splits):
        train_dates = master_prices.index[train_bounds]
        test_dates = master_prices.index[test_bounds]
        
        train_data_x, train_data_y = master_prices[stock_x].loc[train_dates], master_prices[stock_y].loc[train_dates]
        test_data_x, test_data_y = master_prices[stock_x].loc[test_dates], master_prices[stock_y].loc[test_dates]

        # FIX 2: Seed the Train phase with OLS beta to prevent 6 months of wild, garbage Z-scores
        # x_const = sm.add_constant(np.log(train_data_x))
        # ols_res = sm.OLS(np.log(train_data_y), x_const).fit()
        # train_init_beta = float(ols_res.params.iloc[1])

        # Optimization Phase (Train)
        train_kf_results = KF_Indicator.run(
            train_data_x, train_data_y, Q=fixed_q, R=fixed_r # Pass the converged seed here!
        )

        # Generate thresholds for the optimizer
        _, train_thresholds = generate_hmm_position_sizes(
            train_kf_results.Z_Scores, 
            base_notional=allocation_per_pair, 
            window=126,
            high_vol_thresh=3.5, 
            low_vol_thresh=5.0
        )

        corr_pass, _ = check_correlation(train_data_x, train_data_y, min_corr=0.6)
        coint_pass, _ = test_coint_prices(train_data_x, train_data_y, p_value_thresh=0.05)
        trade_this_split = corr_pass and coint_pass
        
        train_signals = Signal_Indicator.run(
            train_kf_results.Z_Scores,
            train_thresholds,
            entry_thresh=entry_test_values,
            exit_thresh=exit_test_values, 
            init_beta=0.0,
            param_product=True,
            execute_kwargs=dict(chunked=True, chunk_len=100)
        )

        train_pair_prices = pd.concat([train_data_x, train_data_y], axis=1)
        train_pair_prices.columns = [stock_x, stock_y]

        long_entries_train = pd.concat([train_signals.short_entries, train_signals.entries], axis=1, keys=[stock_x, stock_y])
        short_entries_train = pd.concat([train_signals.entries, train_signals.short_entries], axis=1, keys=[stock_x, stock_y])
        long_exits_train = pd.concat([train_signals.short_exits, train_signals.exits], axis=1, keys=[stock_x, stock_y])
        short_exits_train = pd.concat([train_signals.exits, train_signals.short_exits], axis=1, keys=[stock_x, stock_y])

        param_names = list(train_signals.entries.columns.names)
        long_entries_train = long_entries_train.sort_index(axis=1, level=param_names)
        long_exits_train = long_exits_train.sort_index(axis=1, level=param_names)
        short_entries_train = short_entries_train.sort_index(axis=1, level=param_names)
        short_exits_train = short_exits_train.sort_index(axis=1, level=param_names)

        train_asset_weights = pd.DataFrame({stock_x: train_kf_results.Betas, stock_y: 1.0}, index=train_data_y.index)
        train_total_weight = train_asset_weights[stock_x].abs() + train_asset_weights[stock_y].abs()
        train_asset_weights[stock_x] = train_asset_weights[stock_x].abs() / train_total_weight
        train_asset_weights[stock_y] = train_asset_weights[stock_y].abs() / train_total_weight
        
        # Keep train_sizes as a clean, flat 2D dataframe. VectorBT's internal 
        # broadcaster will align this to the 3D signal matrices automatically!
        train_sizes = train_asset_weights * allocation_per_pair

        # FIX: Optimize using dollar values, not shares, to mimic OOS behavior
        train_portfolios = vbt.Portfolio.from_signals(
            train_pair_prices,
            entries=long_entries_train, exits=long_exits_train,
            short_entries=short_entries_train, short_exits=short_exits_train,
            size=train_sizes, size_type='Value', init_cash=allocation_per_pair, slippage=slippage,
            cash_sharing=True, group_by=param_names, fees=0.0001, freq='1D', accumulate=False 
        )

        best_idx = train_portfolios.sharpe_ratio().idxmax()
        best_entry, best_exit = best_idx[0], best_idx[1]

        # Bench the pair if the optimizer cannot find a profitable threshold after fees
        best_sharpe = train_portfolios.sharpe_ratio().max()
        if pd.isna(best_sharpe) or best_sharpe <= 0:
            trade_this_split = False

        if trade_this_split:
            trade_this_cycle_overall = True

        # FIX 2: Immediately destroy the massive training matrices and flush RAM
        del train_portfolios
        del train_signals
        del long_entries_train, short_entries_train, long_exits_train, short_exits_train
        gc.collect()

        # Out of Sample Phase (Test)
        # FIX: Ensure explicitly cast raw floats to prevent initialization type errors
        test_kf_results = KF_Indicator.run(
            test_data_x, test_data_y, Q=fixed_q, R=fixed_r,
            init_beta=float(train_kf_results.s_beta.iloc[-1]), init_alpha=float(train_kf_results.s_alpha.iloc[-1]),
            init_c00=float(train_kf_results.c00.iloc[-1]), init_c01=float(train_kf_results.c01.iloc[-1]),
            init_c10=float(train_kf_results.c10.iloc[-1]), init_c11=float(train_kf_results.c11.iloc[-1])
        )

        pair_prices = pd.concat([test_data_x, test_data_y], axis=1)
        pair_prices.columns = [stock_x, stock_y]

        padded_z_scores = pd.concat([train_kf_results.Z_Scores.iloc[-126:], test_kf_results.Z_Scores])
        
        dynamic_test_sizes, dynamic_test_thresholds = generate_hmm_position_sizes(
            padded_z_scores, 
            base_notional=allocation_per_pair, 
            window=126,
            high_vol_weight=0.5, 
            low_vol_weight=1.0*leverage,
            high_vol_thresh=3.5, # Tighter stop in chaotic markets
            low_vol_thresh=5.0   # Wider leash in calm markets
        )
        
        dynamic_test_sizes = dynamic_test_sizes.loc[test_dates]
        dynamic_test_thresholds = dynamic_test_thresholds.loc[test_dates]

        test_signals = Signal_Indicator.run(
            test_kf_results.Z_Scores, 
            dynamic_test_thresholds, # <-- Pass the threshold array
            entry_thresh=best_entry, 
            exit_thresh=best_exit
        )

        long_entries = pd.DataFrame({stock_x: test_signals.short_entries, stock_y: test_signals.entries}, index=test_data_y.index)
        short_entries = pd.DataFrame({stock_x: test_signals.entries, stock_y: test_signals.short_entries}, index=test_data_y.index)
        long_exits = pd.DataFrame({stock_x: test_signals.short_exits, stock_y: test_signals.exits}, index=test_data_y.index)
        short_exits = pd.DataFrame({stock_x: test_signals.exits, stock_y: test_signals.short_exits}, index=test_data_y.index)

        asset_weights = pd.DataFrame({stock_x: test_kf_results.Betas, stock_y: 1.0}, index=test_data_y.index)
        total_weight = asset_weights[stock_x].abs() + asset_weights[stock_y].abs()
        asset_weights[stock_x] = asset_weights[stock_x].abs() / total_weight
        asset_weights[stock_y] = asset_weights[stock_y].abs() / total_weight

        if trade_this_split:
            final_test_sizes = asset_weights.multiply(dynamic_test_sizes, axis=0)
        else:
            final_test_sizes = pd.DataFrame(0.0, index=asset_weights.index, columns=asset_weights.columns)

        oos_prices_list.append(pair_prices)
        oos_long_entries_list.append(long_entries)
        oos_short_entries_list.append(short_entries)
        oos_long_exits_list.append(long_exits)
        oos_short_exits_list.append(short_exits)
        oos_sizes_list.append(final_test_sizes)
        oos_z_scores_list.append(test_kf_results.Z_Scores)

    if not oos_prices_list:
        # Return None or a specific error flag so the main process can skip this pair
        return None
    
    # Compile Master OOS
    master_pair_prices = pd.concat(oos_prices_list)
    master_long_entries = pd.concat(oos_long_entries_list)
    master_short_entries = pd.concat(oos_short_entries_list)
    master_long_exits = pd.concat(oos_long_exits_list)
    master_short_exits = pd.concat(oos_short_exits_list)
    master_sizes = pd.concat(oos_sizes_list)
    master_z_scores = pd.concat(oos_z_scores_list)

    valid_idx = ~master_pair_prices.index.duplicated(keep='first')
    multi_cols = pd.MultiIndex.from_tuples(
        [(pair_name, stock_x), (pair_name, stock_y)], 
        names=['Pair', 'Ticker'] 
    )
    
    # --- RIGID INDEX CONSISTENCY BLOCK ---
    def finalize_df(df, cols):
        df = df.loc[valid_idx].copy()
        if cols is not None:
            df.columns = cols
        df.index.name = 'date' # Force name
        df.index = pd.to_datetime(df.index).tz_localize(None) # Force type
        return df

    raw_corr = master_pair_prices.loc[valid_idx][stock_x].rolling(window=60).corr(master_pair_prices.loc[valid_idx][stock_y])
    # Wrap correlation in a DataFrame so we can force the index name
    # Format the Series cleanly
    raw_corr.name = pair_name
    raw_corr.index = pd.to_datetime(raw_corr.index).tz_localize(None)
    raw_corr.index.name = 'date'

    return {
        'pair_name': pair_name,
        'prices': finalize_df(master_pair_prices, multi_cols),
        'l_ent': finalize_df(master_long_entries, multi_cols),
        's_ent': finalize_df(master_short_entries, multi_cols),
        'l_ext': finalize_df(master_long_exits, multi_cols),
        's_ext': finalize_df(master_short_exits, multi_cols),
        'sizes': finalize_df(master_sizes, multi_cols),
        'trade_this_split': trade_this_split, # Pass the pass/fail flag
        'trade_this_cycle': trade_this_cycle_overall, # Pass the pass/fail flag
        'rolling_corr': raw_corr
    }


# --- REPLACED MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    df_pairs = pd.read_csv("SP500_Cointegrated_Pairs_10Yr.csv")
    df_pairs = df_pairs.sort_values(by='Correlation', ascending=False)
    filtered_df = df_pairs[(df_pairs['Correlation'] >= 0.90) & (df_pairs['Coint_P_Value'] < 0.05)]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datas_dir = os.path.join(base_dir, 'datas')
    
    # Change '4' to a smaller number (like 1.5) if you want to use your 483-day data
    years_per_window = 4 
    window_len = int(252 * years_per_window) 
    
    # 2. Set the filter to match that requirement exactly
    REQUIRED_DAYS = window_len 

    candidate_pairs = []
    print(f"Filtering pairs. Minimum history required for strategy: {REQUIRED_DAYS} days...")

    for row in filtered_df[['Leg_1', 'Leg_2']].itertuples(index=False):
        f_x = os.path.join(datas_dir, f"{row.Leg_1}_Au.csv")
        f_y = os.path.join(datas_dir, f"{row.Leg_2}_Au.csv")
        
        if os.path.exists(f_x) and os.path.exists(f_y):
            # Get the actual length of the files
            len_x = len(pd.read_csv(f_x, usecols=[0]))
            len_y = len(pd.read_csv(f_y, usecols=[0]))
            actual_history = min(len_x, len_y)
            
            if actual_history >= REQUIRED_DAYS:
                candidate_pairs.append((row.Leg_1, row.Leg_2))
            
        if len(candidate_pairs) == 165:
            break

    print(f"Found {len(candidate_pairs)} pairs with enough history ({REQUIRED_DAYS}+ days).")
    # FIX: Use 'outer' join to prevent row-deletion, then ffill/dropna
    master_prices = load_data_for_pairs(candidate_pairs) 
    master_prices = master_prices.ffill().dropna() 

    total_days = len(master_prices)
    window_len = 252 * 4
    if total_days <= window_len:
        print(f"❌ ERROR: Combined data only has {total_days} days. Need {window_len}+ for walk-forward.")
        exit()

    step_size = 252
    start_idxs = np.arange(0, total_days - window_len + 1, step_size)
    end_idxs = start_idxs + window_len
    first_oos_date = master_prices.index[end_idxs[0]]

    splits = list(RangeSplitter().split(X=master_prices.index, start_idxs=start_idxs, end_idxs=end_idxs, set_lens=(0.75,)))

    # Parameters
    MASTER_PORTFOLIO_CAPITAL = 600000
    MAX_PAIRS = 40
    LEVERAGE = 1
    SLIPPAGE = 0.0
    ALLOCATION_PER_PAIR = (MASTER_PORTFOLIO_CAPITAL / MAX_PAIRS)
    fixed_q, fixed_r = 1e-5, 1e-4
    entry_test_values = np.arange(1, 2.5, 0.20)
    exit_test_values = np.arange(0, 0.3, 0.05)

    # Initialize Lists
    portfolio_prices_list, portfolio_sizes_list = [], []
    portfolio_l_ent_list, portfolio_s_ent_list = [], []
    portfolio_l_ext_list, portfolio_s_ext_list = [], []
    portfolio_rolling_corr_dict = {}
    active_pair_count = 0
    processed_count = 0

    optimal_workers = 3
    print(f"\nDeploying {optimal_workers} cores. Target: {MAX_PAIRS} valid pairs...")
    print(f"Total Inital Portfolio Value | ${MASTER_PORTFOLIO_CAPITAL:.2f}")
    print(f"Allocation to Each Pair | ${ALLOCATION_PER_PAIR:.2f}")
    print(f"Leverage | {LEVERAGE}x")
    # --- ADVANCED TQDM PROGRESS BAR ---
    # Customize the exact layout: Description | Bar | X/Y [Time] Postfix
    custom_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"

    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        future_to_pair = {
            executor.submit(
                process_single_pair, p, splits, master_prices, None, 
                entry_test_values, exit_test_values, ALLOCATION_PER_PAIR, fixed_q, fixed_r, LEVERAGE, SLIPPAGE
            ): p for p in candidate_pairs
        }
        
        # Initialize the tqdm object explicitly
        with tqdm(total=len(candidate_pairs), 
                  desc="Running Walk-Forward Analysis", 
                  bar_format=custom_format, 
                  colour="#00ff9d", # Cyber-green to match your Plotly theme
                  smoothing=0.1) as pbar:
            
            for future in concurrent.futures.as_completed(future_to_pair):
                try: 
                    res = future.result()
                    processed_count += 1
                    
                    if res is not None:
                        pair_name = res['pair_name']
                        
                        if res['trade_this_cycle'] and active_pair_count < MAX_PAIRS:
                            portfolio_prices_list.append(res['prices'])
                            portfolio_l_ent_list.append(res['l_ent'])
                            portfolio_s_ent_list.append(res['s_ent'])
                            portfolio_l_ext_list.append(res['l_ext'])
                            portfolio_s_ext_list.append(res['s_ext'])
                            portfolio_sizes_list.append(res['sizes'])
                            portfolio_rolling_corr_dict[res['pair_name']] = res['rolling_corr']
                            active_pair_count += 1
                            
                            # NEW: Calculate rate and update postfix
                            acceptance_rate = (active_pair_count / processed_count) * 100
                            pbar.set_postfix_str(f"✅ Accepted: {pair_name} | Portfolio: {active_pair_count}/{MAX_PAIRS} | Rate: {acceptance_rate:.1f}%")
                        else:
                            # NEW: Calculate rate and update postfix
                            acceptance_rate = (active_pair_count / processed_count) * 100
                            pbar.set_postfix_str(f"❌ Rejected: {pair_name} | Portfolio: {active_pair_count}/{MAX_PAIRS} | Rate: {acceptance_rate:.1f}%")
                    else:
                         # NEW: Calculate rate and update postfix
                         acceptance_rate = (active_pair_count / processed_count) * 100
                         pbar.set_postfix_str(f"⚠️ Skipped: Insufficient Data | Portfolio: {active_pair_count}/{MAX_PAIRS} | Rate: {acceptance_rate:.1f}%")

                    pbar.update(1)
                        
                except Exception as e:
                    pbar.write(f"Worker Error: {e}")
                    break

    print(f"\nFinal Portfolio Size: {active_pair_count} pairs.")

    # --- AGGREGATION & PLOTTING ---
    print("\n==========================================")
    print("=== GENERATING PORTFOLIO-LEVEL METRICS ===")
    # 1. Concatenate all pair signals into massive DataFrames
    all_prices = pd.concat(portfolio_prices_list, axis=1).loc[first_oos_date:]
    all_l_ent = pd.concat(portfolio_l_ent_list, axis=1).loc[first_oos_date:]
    all_s_ent = pd.concat(portfolio_s_ent_list, axis=1).loc[first_oos_date:]
    all_l_ext = pd.concat(portfolio_l_ext_list, axis=1).loc[first_oos_date:]
    all_s_ext = pd.concat(portfolio_s_ext_list, axis=1).loc[first_oos_date:]
    all_sizes = pd.concat(portfolio_sizes_list, axis=1).loc[first_oos_date:]
    all_rolling_corr_df = pd.DataFrame(portfolio_rolling_corr_dict).loc[first_oos_date:]

    # 2. Build the True Capital Sweeping Master Portfolio
    oos_exit_mask = all_l_ext | all_s_ext
    all_sizes = all_sizes.mask(oos_exit_mask, np.inf)

    # FIX: Transpose (.T), group by the pair name, find any active trades, then transpose back and sum
    active_pairs_per_day = (all_sizes != 0).T.groupby(level=0).any().T.sum(axis=1)

    # By setting cash_sharing=True and group_by=True, VectorBT pools all $500k into ONE account.
    master_portfolio = vbt.Portfolio.from_signals(
        all_prices,
        entries=all_l_ent, exits=all_l_ext,
        short_entries=all_s_ent, short_exits=all_s_ext,
        size=all_sizes, size_type='Value', 
        init_cash=MASTER_PORTFOLIO_CAPITAL,  # The full $500k is available globally
        cash_sharing=True, group_by=True, slippage=SLIPPAGE,
        fees=0.0001, freq='1D', accumulate=False 
    )

    # 3. Create a secondary un-grouped portfolio solely to generate the individual equity charts
    indiv_portfolio = vbt.Portfolio.from_signals(
        all_prices, entries=all_l_ent, exits=all_l_ext, short_entries=all_s_ent, short_exits=all_s_ext,
        size=all_sizes, size_type='Value', init_cash=ALLOCATION_PER_PAIR,  
        cash_sharing=True, group_by=all_prices.columns.get_level_values(0), 
        slippage=SLIPPAGE, fees=0.0001, freq='1D', accumulate=False 
    )
    
    # 4. Extract metrics cleanly to match your existing HTML plotting logic
    all_values_df = indiv_portfolio.value()
    aggregate_returns = master_portfolio.returns() 
    vbt_returns = aggregate_returns.vbt.returns(freq='1D')
    overall_portfolio_value = master_portfolio.value()

    # Get the final equity value for every traded pair
    final_pair_values = all_values_df.iloc[-1]
    
    # Count how many finished with more money than they started with
    profitable_pairs_count = (final_pair_values > ALLOCATION_PER_PAIR).sum()
    total_traded_pairs = len(final_pair_values)
    
    # Calculate the percentage
    if total_traded_pairs > 0:
        oos_profitable_pct = (profitable_pairs_count / total_traded_pairs) * 100
    else:
        oos_profitable_pct = 0.0

    # --- Value at Risk (VaR) & CVaR Calculation ---
    confidence_level = 0.99
    
    # Historical VaR (Empirical 5th Percentile)
    # Sorts all daily returns and finds the exact cutoff for the worst 5% of days
    historical_var_pct = aggregate_returns.quantile(1 - confidence_level)
    historical_var_dollars = MASTER_PORTFOLIO_CAPITAL * historical_var_pct

    # Conditional VaR (Expected Shortfall)
    # The average of all daily returns that fall BEYOND the Historical VaR threshold
    cvar_pct = aggregate_returns[aggregate_returns <= historical_var_pct].mean()
    cvar_dollars = MASTER_PORTFOLIO_CAPITAL * cvar_pct

    print(f"Cumulative Return:          {vbt_returns.total() * 100:.2f}%")
    print(f"Max Drawdown:               {vbt_returns.max_drawdown() * 100:.2f}%")
    print(f"Sharpe Ratio:               {vbt_returns.sharpe_ratio():.2f}")
    print(f"Profitable Pairs (%):       {oos_profitable_pct:.2f}%")
    print(f"Historical Daily VaR (99%): {historical_var_pct * 100:.2f}% (${abs(historical_var_dollars):,.2f})")
    print(f"Conditional VaR (CVaR):     {cvar_pct * 100:.2f}% (${abs(cvar_dollars):,.2f})")

    # --- BENCHMARK COMPARISON (S&P 500) ---
    print("\nFetching benchmark data (S&P 500) for comparison plot...")
    # Add a 1-day buffer to the end date to ensure the final day is captured
    start_dt = all_prices.index[0]
    end_dt = all_prices.index[-1] + pd.Timedelta(days=1)
    
    bench_tickers = ['^GSPC', 'MNA', 'SGOV']

    bench_df = yf.download(bench_tickers, start=start_dt, end=end_dt)

    
    # Handle yfinance >= 0.20 MultiIndex changes dynamically
    if isinstance(bench_df.columns, pd.MultiIndex):
        bench_close = bench_df['Close']
    else:
        bench_close = bench_df[['Close']] # Fallback if single column behavior changes
        
    # Reindex to exactly match the backtest's trading days and forward fill any missing data
    bench_close = bench_close.reindex(all_prices.index).ffill()
    
    # Calculate geometric cumulative returns for all benchmarks simultaneously
    bench_cum_returns = (1 + bench_close.pct_change().fillna(0)).cumprod() - 1
    strat_cum_returns = (1 + aggregate_returns.fillna(0)).cumprod() - 1

    # --- PLOT GENERATION ---
    standard_layout = dict(
        template='plotly_white',        # Uniform height for all charts
        height=550,
        width=1400,
        autosize=True,       # Let it fill the HTML card width
        margin=dict(t=40, b=20, l=20, r=20)
    )

    drawdown_plot = vbt_returns.drawdowns.plot()
    drawdown_plot.update_layout(title='Aggregate Portfolio Drawdowns', template='plotly_white', width=1300)
    
    overall_val_plot = go.Figure()
    overall_val_plot.add_trace(go.Scatter(x=overall_portfolio_value.index, y=overall_portfolio_value.values, name='Total Portfolio Equity', line=dict(color="#0077ff", width=2.5)))
    overall_val_plot.update_layout(title='Portfolio Equity Curve', template='plotly_white', width=1300)

    individual_val_plot = go.Figure()
    for pair_name in all_values_df.columns:
        individual_val_plot.add_trace(go.Scatter(x=all_values_df.index, y=all_values_df[pair_name], mode='lines', name=f'{pair_name} Equity', line=dict(width=1.5)))
    individual_val_plot.update_layout(title='Equity Curve per Individual Spread', template='plotly_white', width=1300)

    # NEW: Rolling Correlation Heatmap (Time vs Pairs)
    corr_heatmap_plot = go.Figure(data=go.Heatmap(
        z=all_rolling_corr_df.T.values,
        x=all_rolling_corr_df.index,
        y=all_rolling_corr_df.columns,
        colorscale='RdBu',  # Red = Negative, Blue = Positive
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))

    corr_heatmap_plot.update_layout(
        title='Rolling Correlation Heatmap over Time',
        xaxis_title='Date',
        yaxis_title='Traded Pairs',
        yaxis=dict(
            tickmode='linear',
            dtick=1,              # Forces every single pair label to display
            tickfont=dict(size=9) # Shrinks the text so all 40+ fit in the container
        ),
        template='plotly_white',
        width=1300
    )

    capacity_plot = go.Figure(go.Scatter(
        x=active_pairs_per_day.index, 
        y=active_pairs_per_day.values, 
        fill='tozeroy', 
        name='Active Pairs',
        line=dict(color='#3498db')
    ))
    capacity_plot.update_layout(title='Portfolio Capacity (Active Pairs)', template='plotly_white', width=1300)

    # UPDATED: Multi-Benchmark Cumulative Returns Plot
    cum_bench_plot = go.Figure()
    # Strategy trace (Your Algorithm)
    cum_bench_plot.add_trace(go.Scatter(x=strat_cum_returns.index, y=strat_cum_returns.values * 100, name='Strategy Cumulative Returns', line=dict(color='#00ff9d', width=3.5)))
    # S&P 500
    cum_bench_plot.add_trace(go.Scatter(x=bench_cum_returns.index, y=bench_cum_returns['^GSPC'].values * 100, name='S&P 500', line=dict(color='#f39c12', width=2)))
    # Market Neutral ETF
    cum_bench_plot.add_trace(go.Scatter(x=bench_cum_returns.index, y=bench_cum_returns['MNA'].values * 100, name='Market Neutral (MNA)', line=dict(color='#3498db', width=2)))    
    # Risk-Free Proxy
    cum_bench_plot.add_trace(go.Scatter(x=bench_cum_returns.index, y=bench_cum_returns['SGOV'].values * 100, name='Short-Term Treasuries (SGOV)', line=dict(color='#9b59b6', width=2)))

    cum_bench_plot.update_layout(
        title='Strategy vs Multi-Asset Benchmarks Cumulative Returns', 
        yaxis_title='Cumulative Return (%)',
        template='plotly_white',
        width=1300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # NEW: Custom Walk-Forward Cross Validation Plot
    cv_plot = go.Figure()

    # Loop through the raw splits list to extract exact start and end dates
    for i, (train_bounds, test_bounds) in enumerate(splits):
        train_start = master_prices.index[train_bounds[0]]
        train_end = master_prices.index[train_bounds[-1]]
        test_start = master_prices.index[test_bounds[0]]
        test_end = master_prices.index[test_bounds[-1]]
        
        # Plot the In-Sample (Train) window
        cv_plot.add_trace(go.Scatter(
            x=[train_start, train_end], 
            y=[f"Split {i+1}", f"Split {i+1}"],
            mode='lines',
            line=dict(color='#3498db', width=16),
            name='Train (In-Sample)' if i == 0 else None,
            showlegend=True if i == 0 else False
        ))
        
        # Plot the Out-Of-Sample (Test) window
        cv_plot.add_trace(go.Scatter(
            x=[test_start, test_end], 
            y=[f"Split {i+1}", f"Split {i+1}"],
            mode='lines',
            line=dict(color='#f39c12', width=16),
            name='Test (Out-Of-Sample)' if i == 0 else None,
            showlegend=True if i == 0 else False
        ))

    # Apply the standard layout and reverse the Y-axis so Split 1 is at the top
    cv_plot.update_layout(
        title='Walk-Forward Optimization Splits (In-Sample vs Out-Of-Sample)',
        yaxis=dict(autorange="reversed"),
        template='plotly_white',
        height=550,
        width=1300 
    )

    figures = [
        {"title": "Total Portfolio Equity Curve", "figure": overall_val_plot},
        {"title": "Strategy vs S&P 500 Cumulative Returns", "figure": cum_bench_plot},
        {"title": "Walk-Forward Optimization Splits", "figure": cv_plot},
        {"title": "Individual Spread Equity Curves", "figure": individual_val_plot},
        {"title": "Rolling Correlation Matrix", "figure": corr_heatmap_plot}, # Injected Heatmap
        {"title": "Aggregate Portfolio Drawdowns", "figure": drawdown_plot},
        {"title": "Portfolio Trading Capacity", "figure": capacity_plot}
    ]

    html_filename = "Optimized_Kalman_Pairs_Dashboard.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Pairs Trading Dashboard</title><style>body { font-family: Arial; background: #121212; color: #fff; padding: 20px; } .card { background: #1e1e1e; padding: 20px; margin-bottom: 30px; border-radius: 8px; }</style></head><body>")
        for i, item in enumerate(figures):
            f.write(f'<div class="card"><h2>{item["title"]}</h2>{item["figure"].to_html(full_html=False, include_plotlyjs="cdn" if i==0 else False)}</div>')
        f.write("</body></html>")

    print(f"Fast Backtest Complete. Dashboard saved to {html_filename}")


