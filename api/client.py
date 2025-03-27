import time
import logging
from kiteconnect import KiteConnect
from api.auth import get_access_token
from config.settings import MAX_API_CALLS_PER_MINUTE, API_CALL_INTERVAL, INSTRUMENT_CACHE_DURATION

# Initialize logger
logger = logging.getLogger("TradingBot")


class KiteClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.access_token = None
        self.last_api_call_time = 0
        self.instrument_cache = {"cache_time": 0, "data": {}}

    def initialize(self):
        """Initialize the KiteConnect client with valid access token"""
        if self.kite is None:
            self.access_token = get_access_token(self.api_key, self.api_secret)
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.info("KiteConnect client initialized successfully")
        return self.kite

    def rate_limited_api_call(self, func, *args, **kwargs):
        """
        Wrapper for API calls with rate limiting and retry logic.

        Args:
            func: KiteConnect API function to call
            *args, **kwargs: Arguments to pass to the function

        Returns:
            The result of the API call
        """
        # Ensure minimum time between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time

        if time_since_last_call < API_CALL_INTERVAL:
            sleep_time = API_CALL_INTERVAL - time_since_last_call
            time.sleep(sleep_time)

        # Update last call time
        self.last_api_call_time = time.time()

        # Attempt the API call with retry logic
        max_retries = 5
        retry_delay = 2  # starting delay in seconds

        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if "Too many requests" in str(e) and attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    # If it's not a rate limit issue or we're out of retries, re-raise
                    raise

    def get_cached_instruments(self, exchange):
        """
        Get instruments with caching to reduce API calls.

        Args:
            exchange: Exchange code (e.g., "NSE", "NFO")

        Returns:
            List of instruments for the exchange
        """
        current_time = time.time()
        cache_age = current_time - self.instrument_cache["cache_time"]

        # Check if we have a valid cache for this exchange
        if exchange in self.instrument_cache["data"] and cache_age < INSTRUMENT_CACHE_DURATION:
            logger.debug(f"Using cached instruments for {exchange}")
            return self.instrument_cache["data"][exchange]

        # Fetch fresh data with rate limiting
        logger.info(f"Fetching fresh instruments for {exchange}")
        try:
            instruments = self.rate_limited_api_call(self.kite.instruments, exchange)

            # Update cache
            self.instrument_cache["data"][exchange] = instruments
            self.instrument_cache["cache_time"] = current_time

            return instruments
        except Exception as e:
            logger.error(f"Error fetching instruments for {exchange}: {e}")

            # If we have stale cache data, better to use it than nothing
            if exchange in self.instrument_cache["data"]:
                logger.warning(f"Using stale cached instruments for {exchange}")
                return self.instrument_cache["data"][exchange]

            return []

    def get_instrument_token(self, exchange, symbol):
        """
        Get the instrument token for a given symbol.

        Args:
            exchange: Exchange code (e.g., "NSE", "NFO")
            symbol: Symbol string (e.g., "TCS", "TCS-EQ", "NSE:TCS")

        Returns:
            Instrument token or None if not found
        """
        try:
            instruments = self.get_cached_instruments(exchange)
            if not instruments:
                logger.error(f"No instruments available for {exchange}")
                return None
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return None

        # Manual symbol cleaning
        symbol_clean = symbol.strip().upper()

        # Remove exchange prefix if present
        if ':' in symbol_clean:
            symbol_clean = symbol_clean.split(':', 1)[1]

        # Remove -EQ suffix if present
        clean_for_match = symbol_clean
        if symbol_clean.endswith('-EQ'):
            clean_for_match = symbol_clean[:-3]

        logger.debug(f"Looking for instrument token for: {symbol_clean} (matching: {clean_for_match})")

        # Find the matching instrument - first try exact match
        for inst in instruments:
            inst_symbol = inst["tradingsymbol"].strip().upper()
            # Try both with and without -EQ suffix
            if inst_symbol == symbol_clean or inst_symbol == clean_for_match:
                logger.debug(f"Found token {inst['instrument_token']} for {symbol_clean}")
                return inst["instrument_token"]

        # If exact match fails, try more flexible matching
        for inst in instruments:
            inst_symbol = inst["tradingsymbol"].strip().upper()
            # Try matching the start of the symbol
            if inst_symbol.startswith(clean_for_match):
                logger.debug(f"Found token {inst['instrument_token']} for {symbol_clean} via partial match")
                return inst["instrument_token"]

        logger.warning(f"No instrument token found for {symbol_clean}")
        return None

    # Wrapper methods for common API calls with rate limiting
    def quote(self, symbols):
        """Get market quotes for symbols"""
        return self.rate_limited_api_call(self.kite.quote, symbols)

    def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False):
        """Get historical data for an instrument"""
        return self.rate_limited_api_call(
            self.kite.historical_data,
            instrument_token, from_date, to_date, interval, continuous
        )

    def margins(self):
        """Get user margins"""
        return self.rate_limited_api_call(self.kite.margins)

    def orders(self):
        """Get order history"""
        return self.rate_limited_api_call(self.kite.orders)

    def positions(self):
        """Get user positions"""
        return self.rate_limited_api_call(self.kite.positions)

    def place_order(self, **params):
        """Place an order"""
        return self.rate_limited_api_call(self.kite.place_order, **params)

    def cancel_order(self, order_id, variety):
        """Cancel an order"""
        return self.rate_limited_api_call(self.kite.cancel_order, order_id=order_id, variety=variety)

    def place_gtt(self, **params):
        """Place a GTT order"""
        return self.rate_limited_api_call(self.kite.place_gtt, **params)

    def get_gtts(self):
        """Get all GTT orders"""
        return self.rate_limited_api_call(self.kite.get_gtts)

    def delete_gtt(self, trigger_id):
        """Delete a GTT order"""
        return self.rate_limited_api_call(self.kite.delete_gtt, trigger_id=trigger_id)