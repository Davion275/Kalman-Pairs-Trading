import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

IBKR_HOST = os.getenv('IBKR_HOST')
IBKR_PORT = os.getenv('IBKR_PORT')
IBKR_CLIENT_ID = os.getenv('IBKR_CLIENT_ID')

# PAIRS = [('V','MA'), ('GOOGL', 'GOOG'),('XOM', 'CVX')]
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'

# Strategy Parameters
TRADE_QTY = 100
ROLLING_WINDOW = 60
BAR_SIZE = '1 hour'
DURATION_STR = '1 D'

# Kalman Parameters
Q_VAR = 1e-5  # Process Noise
R_VAR = 1e-4  # Measurement Noise
ENTRY_THRESHOLD = 2.0  # Z-score to enter trade
EXIT_THRESHOLD = 0.0   # Z-score to exit