# --- IMPORTANT: Set these BEFORE importing numpy or statsmodels ---
# This stops the underlying math libraries from aggressively multithreading 
# inside our already multithreaded Python workers (prevents "Thread Thrashing").
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint
import itertools
import concurrent.futures
from tqdm import tqdm
import math

# --- 1. FETCH UNIVERSE & DATA ---
def get_sp500_tickers():
    print("Fetching S&P 500 tickers...")
    df = pd.read_csv('constituents.csv')
    tickers = df['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def download_historical_data(tickers, period="10y"):
    print(f"Downloading {period} of historical data for {len(tickers)} tickers...")
    data = yf.download(tickers, period=period, progress=False)['Close']
    thresh = int(len(data) * 0.95) 
    data = data.dropna(axis=1, thresh=thresh)
    data = data.ffill().bfill()
    return data

# --- 2. STATISTICAL TESTING ---
worker_price_arrays = {}

def init_worker(shared_arrays):
    global worker_price_arrays
    worker_price_arrays = shared_arrays

def test_single_pair(pair, min_corr=0.85, p_value_thresh=0.05):
    stock_x, stock_y = pair
    arr_x = worker_price_arrays[stock_x]
    arr_y = worker_price_arrays[stock_y]
    
    corr = np.corrcoef(np.log(arr_x), np.log(arr_y))[0, 1]
    if corr < min_corr:
        return None
        
    try:
        score, p_value, _ = coint(arr_x, arr_y)
        if p_value < p_value_thresh:
            return {
                'Leg_1': stock_x,
                'Leg_2': stock_y,
                'Correlation': corr,
                'Coint_P_Value': p_value,
                'T_Statistic': score
            }
    except:
        pass
        
    return None

# --- 3. MAIN EXECUTION ---
if __name__ == '__main__':
    # Parameters
    YEARS = "10y"
    MIN_CORR = 0.85
    P_VALUE_THRESH = 0.05
    CORES_TO_USE = 4  # Explicit limit 
    
    # 1. Setup Data
    tickers = get_sp500_tickers()
    price_data = download_historical_data(tickers, period=YEARS)
    
    valid_tickers = price_data.columns.tolist()
    all_pairs = list(itertools.combinations(valid_tickers, 2))
    
    print("Optimizing data structures for multiprocessing...")
    price_arrays = {ticker: price_data[ticker].values for ticker in valid_tickers}
    
    print(f"Data prepared. Testing {len(all_pairs)} pairs across {CORES_TO_USE} CPU cores...")

    # 2. Execute Multiprocessing Pool in 10% Increments
    valid_pairs = []
    total_tasks = len(all_pairs)
    
    # Calculate exactly how many tasks make up 10%
    batch_size = math.ceil(total_tasks / 10)
    
    # max_workers physically limits how many CPU cores are engaged
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=CORES_TO_USE,
        initializer=init_worker, 
        initargs=(price_arrays,)
    ) as executor:
        
        # Loop 10 times to process each 10% batch sequentially
        for i in range(10):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_tasks)
            current_batch = all_pairs[start_idx:end_idx]
            
            if not current_batch:
                break
                
            print(f"\n--- Queuing Batch {i+1}/10 ({len(current_batch)} pairs) ---")
            
            # Map the current 10% chunk
            results = executor.map(
                test_single_pair, 
                current_batch, 
                chunksize=500 # Forces workers to take 500 tasks at a time, protecting RAM
            )
            
            # Update the progress bar specifically for this batch
            for result in tqdm(results, total=len(current_batch), desc=f"Batch {i+1} Progress"):
                if result is not None:
                    valid_pairs.append(result)

    # 3. Process and Export Results
    results_df = pd.DataFrame(valid_pairs)
    
    if not results_df.empty:
        results_df = results_df.sort_values(by='Coint_P_Value').reset_index(drop=True)
        print(f"\n✅ Scan Complete! Found {len(results_df)} highly cointegrated pairs.")
        
        output_file = "SP500_Cointegrated_Pairs_10Yr.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("\nScan Complete. No pairs met the strict criteria.")