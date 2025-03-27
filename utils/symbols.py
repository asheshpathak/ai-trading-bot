import os
import logging
from config.settings import SYMBOLS_DIR

# Initialize logger
logger = logging.getLogger("TradingBot")


def clean_symbol(symbol, remove_exchange=True, remove_suffix=True):
    """
    Cleans a trading symbol by optionally removing exchange prefix and suffix.

    Args:
        symbol (str): The symbol to clean (e.g. 'NSE:TCS-EQ' or 'TCS-EQ' or 'TCS')
        remove_exchange (bool): Whether to remove exchange prefix (e.g. 'NSE:')
        remove_suffix (bool): Whether to remove suffix (e.g. '-EQ')

    Returns:
        str: Cleaned symbol
    """
    symbol = symbol.strip().upper()

    # Remove exchange prefix if present and requested
    if remove_exchange and ':' in symbol:
        symbol = symbol.split(':', 1)[1]

    # Remove suffix if present and requested
    if remove_suffix and symbol.endswith('-EQ'):
        symbol = symbol[:-3]

    return symbol


def read_symbols_from_file(filename=None):
    """
    Reads symbols from a file, either specified or default.

    Args:
        filename: Path to symbols file or None to use default

    Returns:
        list: List of symbols
    """
    # If no filename provided, use default
    if not filename:
        filename = os.path.join(SYMBOLS_DIR, "trading_symbols.txt")

    # Check if file exists
    if not os.path.exists(filename):
        logger.error(f"Symbols file not found: {filename}")
        return []

    try:
        with open(filename, "r") as f:
            content = f.read()

        # Process symbols
        symbols = [s.strip() for s in content.split(",") if s.strip()]
        logger.info(f"Read {len(symbols)} symbols from {filename}")
        return symbols

    except Exception as e:
        logger.error(f"Error reading symbols file {filename}: {e}")
        return []


def save_symbols_to_file(symbols, filename=None):
    """
    Saves a list of symbols to a file.

    Args:
        symbols: List of symbols to save
        filename: Path to save symbols to or None to use default

    Returns:
        bool: True if successful, False otherwise
    """
    # If no filename provided, use default
    if not filename:
        # Ensure directory exists
        os.makedirs(SYMBOLS_DIR, exist_ok=True)
        filename = os.path.join(SYMBOLS_DIR, "trading_symbols.txt")

    try:
        # Prepare symbols as comma-separated string
        symbols_str = ", ".join([s.strip() for s in symbols if s.strip()])

        with open(filename, "w") as f:
            f.write(symbols_str)

        logger.info(f"Saved {len(symbols)} symbols to {filename}")
        return True

    except Exception as e:
        logger.error(f"Error saving symbols to {filename}: {e}")
        return False


def filter_option_symbols(symbols):
    """
    Filters a list of symbols to include only those valid for options trading.

    Args:
        symbols: List of symbols to filter

    Returns:
        list: Filtered symbols list
    """
    # This is a placeholder - in reality, you might check for:
    # - Minimum market cap
    # - Options availability
    # - Liquidity
    # - MWPL availability
    # For now, just return the list as is
    return symbols