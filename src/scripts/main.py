# # --- Live Strategy ---
# # Connect to IBKR
# ib = connect_to_ib()

# # Parameters
# pair = config.PAIRS[2]
# ASSET_A = pair[0] # Dependent Variable (Y)
# ASSET_B = pair[1] # Independent Variable (X)
# QTY = config.TRADE_QTY

# # --- Initialize Kalman Filter ---
# # Only run this once when we first start trading the strategy for the given pair
# # then only use the current beta from new stock prices
# kf = CumulativeReturnsKF(Q=config.Q_VAR, R=config.R_VAR, window=config.ROLLING_WINDOW)
# if kf.state_mean[0] == 0.0:
#     error_history, current_beta = PairsTrading.initialize_kalman(ib, ASSET_A, ASSET_B)


# # Execute Strategy Loop
# current_beta = PairsTrading.pair_strategy(ib, kf, error_history, ASSET_A, ASSET_B, current_beta)
# print(f"Run complete at {datetime.datetime.now()}.")

# # Disconnect from IBKR
# ib.disconnect()


# --- Backtest Strategy ---