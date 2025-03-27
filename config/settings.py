import os
from pathlib import Path

# Project structure
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training")
SYMBOLS_DIR = os.path.join(DATA_DIR, "symbols")
PICKLES_DIR = os.path.join(ROOT_DIR, "pickles")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Ensure directories exist
for dir_path in [DATA_DIR, TRAINING_DATA_DIR, SYMBOLS_DIR, PICKLES_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Exchange settings
EXCHANGE = "NSE"
OPTION_EXCHANGE = "NFO"

# Technical analysis parameters
# ENHANCED: Reverted to traditional values for higher quality signals
RSI_OVERSOLD = 30  # Reverted from relaxed 45
RSI_OVERBOUGHT = 70  # Reverted from relaxed 55

# ENHANCED: Stricter price thresholds for stronger moves
OVERSOLD_THRESHOLD = 0.99  # 1% drop (was 0.995)
OVERBOUGHT_THRESHOLD = 1.01  # 1% rise (was 1.005)

# ENHANCED: Volatility thresholds
MIN_VOLATILITY = 0.01  # Minimum 1% volatility for trade consideration
IDEAL_VOLATILITY = 0.02  # Target volatility level for full position sizing

# ENHANCED: Better risk-reward parameters
BASE_STOPLOSS_PERCENT = 0.04  # Enhanced from 0.03
BASE_TARGET_PERCENT = 0.10  # Enhanced from 0.06

# EMA settings for crossovers
SHORT_EMA = 9
LONG_EMA = 20

# ENHANCED: Time filters to avoid volatile periods
AVOID_MARKET_OPEN_MINS = 15  # Avoid first 15 minutes after market open
AVOID_MARKET_CLOSE_MINS = 15  # Avoid last 15 minutes before market close
PRE_LUNCH_END_HOUR = 11  # End of morning session
POST_LUNCH_START_HOUR = 13  # Start of afternoon session

# ENHANCED: Trend strength parameters
ADX_THRESHOLD = 20  # Minimum ADX for trend confirmation
ADX_STRONG = 30  # Strong trend threshold

# Trading parameters
SCAN_INTERVAL = 60  # seconds between live scans
MIN_MARGIN_REQUIRED = 10000  # Minimum margin required to place a trade

# Risk management
MAX_DAILY_LOSS = 5000
MAX_WEEKLY_LOSS = 15000
MAX_POSITION_SIZE_PCT = 5
MAX_POSITIONS = 5

# API rate limiting
MAX_API_CALLS_PER_MINUTE = 60
API_CALL_INTERVAL = 1.5  # seconds between API calls
INSTRUMENT_CACHE_DURATION = 3600  # Cache instruments for 1 hour

# Order settings
ORDER_TYPE = "NRML"  # Use NRML instead of MIS for overnight positions

# Training parameters
DAYS_OF_HISTORICAL_DATA = 30  # Number of days of historical data to fetch for training

# Logging settings
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL = "DEBUG"
CONSOLE_FORMAT = "standard"  # minimal, standard, or detailed