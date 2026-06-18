import os
import time
import threading
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# --- 1. IBKR API WRAPPER ---
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.bars = []
        self.next_req_id = None
        # Event objects to synchronize asynchronous API calls with our main loop
        self.connection_event = threading.Event()
        self.data_event = threading.Event()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_req_id = orderId
        self.connection_event.set() # Signal that the API is fully connected and ready

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore routine connection messages
        if errorCode not in [2104, 2106, 2158]: 
            print(f"Error {errorCode}: {errorString}")
        # If we hit a pacing violation (162) or missing data (200), skip to next ticker
        if errorCode in [162, 200]:
            self.data_event.set()

    def historicalData(self, reqId, bar):
        # Store bar data in memory
        self.bars.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
   
    def historicalDataEnd(self, reqId, start, end):
        print(f"Finished downloading data for Request ID {reqId}.")
        self.data_event.set() # Signal the main loop to proceed to CSV creation

# --- 2. DATA PREPARATION ---
def get_unique_tickers():
    print("Filtering pairs and extracting unique tickers...")
    df_pairs = pd.read_csv("SP500_Cointegrated_Pairs_10Yr.csv")

    # 1. Add parentheses around the conditions
    filtered_df = df_pairs[(df_pairs['Correlation'] >= 0.90) & (df_pairs['Coint_P_Value'] < 0.05)]

    # 2. Convert to list of tuples
    portfolio_pairs = [tuple(x) for x in filtered_df[['Leg_1', 'Leg_2']].values]
    portfolio_pairs.extend([('V', 'MA'), ('XOM', 'CVX')])

    # 3. Flatten the list of tuples and convert to a Set to remove all duplicates
    unique_tickers = set([ticker for pair in portfolio_pairs for ticker in pair])
    print(f"Extracted {len(unique_tickers)} unique stocks to download.")
    return unique_tickers

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    
    unique_tickers = get_unique_tickers()

    # Establish ONE connection to TWS
    app = IBapi()
    app.connect("127.0.0.1", 7497, 1)

    # Start the API thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Wait for the API to confirm connection
    print("Waiting for IBKR connection...")
    app.connection_event.wait()
    print("API Connected!")

    # Ensure output directory exists (based on your backtesting_vbt.py folder structure)
    os.makedirs('datas', exist_ok=True)

    # Sequentially request historical data
    for ticker in unique_tickers:
        # Check if we already have the data to save time on restarts
        output_file = f"datas/{ticker}_Au.csv"
        if os.path.exists(output_file):
            print(f"{ticker} already exists. Skipping.")
            continue

        print(f"Requesting 10-Year data for {ticker}...")
        
        # Define Contract
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        # Reset data array and event flag
        app.bars = []
        app.data_event.clear()

        # Request 10 Years of Daily data
        app.reqHistoricalData(
            reqId=app.next_req_id, 
            contract=contract, 
            endDateTime="", 
            durationStr="10 Y", 
            barSizeSetting="1 day", 
            whatToShow="TRADES", 
            useRTH=1, 
            formatDate=1, 
            keepUpToDate=False, 
            chartOptions=[]
        )
        
        # PAUSE MAIN LOOP: Wait until historicalDataEnd() is called
        app.data_event.wait()

        # Save to CSV using pandas
        if len(app.bars) > 0:
            df = pd.DataFrame(app.bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(app.bars)} bars to {output_file}")
        
        app.next_req_id += 1
        
        # Crucial: Sleep to respect IBKR's historical pacing violations (max 60 requests per 10 mins)
        time.sleep(5) 

    print("All downloads complete. Disconnecting...")
    app.disconnect()
