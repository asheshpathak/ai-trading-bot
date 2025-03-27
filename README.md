# Trading Bot

A modular, feature-rich automated trading system for options trading on Zerodha.

## Features

- **Technical Analysis**: Implements multiple indicators (RSI, EMA, MACD, VWAP)
- **Machine Learning**: Uses RandomForest models to predict price movements
- **Options Trading**: Automatically selects appropriate option contracts
- **Risk Management**: Position sizing, loss limits, and time-based exits
- **Parallel Processing**: Multi-threaded design for efficient symbol scanning

## Directory Structure

```
trading_bot/
├── config/                 # Configuration settings
│   ├── __init__.py
│   ├── settings.py         # Trading parameters and settings
│   └── credentials.py      # API credentials management
├── api/                    # API interaction
│   ├── __init__.py
│   ├── auth.py             # Authentication with Zerodha
│   ├── client.py           # Rate-limited API client
│   └── data.py             # Market data functions
├── models/                 # Analysis and prediction
│   ├── __init__.py
│   ├── technical.py        # Technical indicators
│   ├── ml_predictor.py     # ML model prediction
│   └── training.py         # Model training
├── trading/                # Trading logic
│   ├── __init__.py
│   ├── signals.py          # Signal generation
│   ├── options.py          # Option selection
│   ├── orders.py           # Order management
│   └── risk.py             # Risk controls
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   └── symbols.py          # Symbol handling
├── data/                   # Data storage
│   ├── symbols/            # Symbol lists
│   │   └── trading_symbols.txt
│   └── training/           # CSV data for training
├── pickles/                # Trained models
├── logs/                   # Log files
├── .env                    # Environment variables (credentials)
├── requirements.txt        # Dependencies
├── main.py                 # Main trading script
└── train.py                # Model training script
```

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone [repository-url]
cd trading_bot

# Create a virtual environment (Python 3.9 recommended)
python -m venv venv

# Activate the environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Credentials Configuration

Set up your Zerodha API credentials using one of these methods:

#### Option 1: Environment Variables (Recommended)

Create a `.env` file in the project root:

```
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
```

#### Option 2: Encrypted Storage

The bot will prompt you to enter credentials on first run and store them securely.

### 3. Trading Symbols

Edit `data/symbols/trading_symbols.txt` to include the symbols you want to trade (comma-separated).

Example:
```
RELIANCE, INFY, TCS, HDFCBANK, ICICIBANK, SBIN
```

### 4. Adjust Settings (Optional)

Review `config/settings.py` to customize:
- Technical indicator parameters
- Risk management settings
- Scan intervals
- Order types

## Usage

### Training ML Models

Before running the bot, you need to train ML models for your symbols:

```bash
# Basic usage
python train.py

# Advanced options
python train.py --symbols path/to/custom_symbols.txt --days 30 --parallel 5 --force
```

**Options:**
- `--symbols`: Path to a custom symbols file (default: data/symbols/trading_symbols.txt)
- `--days`: Days of historical data to fetch (default: 30)
- `--parallel`: Number of parallel workers (default: 5)
- `--force`: Force retraining of existing models

### Running the Trading Bot

```bash
# Start in dry run mode (recommended for testing)
python main.py --dryrun

# Start in live trading mode
python main.py

# Advanced options
python main.py --symbols path/to/custom_symbols.txt --interval 60 --parallel 5 --min-margin 10000 --log-level INFO
```

**Options:**
- `--symbols`: Path to a custom symbols file
- `--interval`: Scan interval in seconds (default: 60)
- `--parallel`: Number of parallel symbol processors (default: 5)
- `--min-margin`: Minimum margin required
- `--log-level`: Logging level (default: INFO)
- `--dryrun`: Run in dry run mode (no actual orders)

## How It Works

### Model Training Process

1. **Data Collection**: Fetches historical minute-data for each symbol
2. **Feature Engineering**: Calculates technical indicators (RSI, EMA, MACD, etc.)
3. **Model Training**: Trains RandomForest classifier for price movement prediction
4. **Model Storage**: Saves models to pickles/ directory for use by the trading bot

### Trading Process

1. **Initialization**:
   - Connects to Zerodha API
   - Loads ML models
   - Initializes components

2. **Market Scanning**:
   - Fetches market data and calculates indicators
   - Generates trading signals using both rule-based and ML models
   - Ranks signals by confidence

3. **Option Selection**:
   - Finds appropriate expiry dates
   - Calculates ATM strike prices
   - Selects option contracts based on signals

4. **Risk Management**:
   - Checks margin requirements
   - Enforces position and loss limits
   - Implements time-based exits

5. **Order Execution**:
   - Places limit orders for entries
   - Sets up GTT orders for stop-loss and target
   - Monitors and manages order statuses

## Logging and Monitoring

The bot maintains separate log files for different types of information:

- **Trading Signals**: `logs/signals_[date].log`
- **Executed Trades**: `logs/trades_[date].log`
- **General Activity**: `logs/trading_bot_[date].log`

## Common Issues and Troubleshooting

### Python Version Compatibility

This project works best with **Python 3.9**. Using newer versions (like Python 3.12) may cause compatibility issues with some dependencies.

If using Python 3.12, you might encounter errors with distutils. Try:
```bash
pip install setuptools
pip install --use-pep517 -r requirements.txt
```

### API Connection Issues

- Verify your API credentials are correct
- Check if your API key has necessary permissions
- Ensure your internet connection is stable

### Data Fetching Problems

- Check if symbols are valid and actively traded
- Verify you have sufficient API rate limits
- Ensure date ranges are valid and not on holidays/weekends

### Order Placement Issues

- Check if you have sufficient margin
- Verify the exchange is open
- Check if the symbol is in circuit limits or suspended

## Best Practices

### Testing

1. **Start with Dry Run**: Always use the `--dryrun` flag first to test without placing real orders
2. **Test Single Symbol**: Start with a single symbol before scaling to many
3. **Component Testing**: Test API connection and data fetching separately

### Production Deployment

1. **Use a Stable Server**: Run on a reliable server with consistent internet connection
2. **Schedule Regular Backups**: Back up trained models and configuration
3. **Implement Monitoring**: Set up alerts for critical errors or unusual behavior
4. **Start Small**: Begin with smaller position sizes and fewer symbols
5. **Regular Verification**: Verify all positions and orders manually

## License

[Specify your license]

## Acknowledgments

- Zerodha for the KiteConnect API
- scikit-learn for machine learning functionality
- [Add other acknowledgments as appropriate]