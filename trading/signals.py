import logging
import concurrent.futures
import pytz
from datetime import datetime, timedelta
from functools import partial
from api.data import fetch_minute_data, fetch_previous_close, fetch_live_quote
from models.technical import compute_enhanced_indicators, evaluate_enhanced_scalping_conditions
from config.settings import (
    MIN_VOLATILITY, ADX_THRESHOLD, ADX_STRONG,
    AVOID_MARKET_OPEN_MINS, AVOID_MARKET_CLOSE_MINS,
    PRE_LUNCH_END_HOUR, POST_LUNCH_START_HOUR
)

# Initialize logger
logger = logging.getLogger("TradingBot")


class SignalGenerator:
    def __init__(self, kite_client, ml_predictor):
        """
        Initialize with KiteClient and MLPredictor instances.

        Args:
            kite_client: KiteClient instance
            ml_predictor: MLPredictor instance
        """
        self.kite_client = kite_client
        self.ml_predictor = ml_predictor
        self.traded_symbols = set()  # Symbols already traded today

    def reset_traded_symbols(self):
        """Reset the set of traded symbols (e.g., at start of new day)"""
        self.traded_symbols.clear()
        logger.info("Traded symbols list has been reset")

    def add_traded_symbol(self, symbol):
        """
        Add a symbol to the traded symbols set.

        Args:
            symbol: Symbol to add
        """
        # Clean symbol for consistent tracking
        clean_symbol = symbol.strip().upper()
        if ":" in clean_symbol:
            clean_symbol = clean_symbol.split(":", 1)[1]
        if clean_symbol.endswith("-EQ"):
            clean_symbol = clean_symbol[:-3]

        self.traded_symbols.add(clean_symbol)
        logger.info(f"Added {clean_symbol} to traded symbols. Will not trade again today.")

    # ENHANCED: Add time-of-day filter
    def time_filter_valid(self, now=None):
        """
        Check if current time passes the time-of-day filters.

        Args:
            now: Current datetime or None to use current time

        Returns:
            tuple: (is_valid, reason)
        """
        if now is None:
            india_tz = pytz.timezone("Asia/Kolkata")
            now = datetime.now(india_tz)

        # Market hours
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # Avoid market open volatility
        avoid_open_until = market_open + timedelta(minutes=AVOID_MARKET_OPEN_MINS)

        # Avoid market close volatility
        avoid_close_from = market_close - timedelta(minutes=AVOID_MARKET_CLOSE_MINS)

        # Avoid lunch hour (optional)
        pre_lunch_end = now.replace(hour=PRE_LUNCH_END_HOUR, minute=30, second=0, microsecond=0)
        post_lunch_start = now.replace(hour=POST_LUNCH_START_HOUR, minute=0, second=0, microsecond=0)

        # Check if current time is valid
        if now < avoid_open_until:
            return False, "Avoiding market open volatility"
        elif now > avoid_close_from:
            return False, "Approaching market close"
        elif pre_lunch_end <= now <= post_lunch_start:
            return False, "Avoiding lunch hour volatility"
        else:
            return True, None

    def process_symbol(self, symbol, symbol_info):
        """
        Process a single symbol and return signal if found.

        Args:
            symbol: Symbol to process
            symbol_info: Dict with symbol info including token and prev_close

        Returns:
            dict: Signal info or None if no signal
        """
        try:
            # Skip symbols already traded today
            clean_symbol = symbol.strip().upper()
            if ":" in clean_symbol:
                clean_symbol = clean_symbol.split(":", 1)[1]
            if clean_symbol.endswith("-EQ"):
                clean_symbol = clean_symbol[:-3]

            if clean_symbol in self.traded_symbols:
                logger.debug(f"Skipping {symbol} - already traded today")
                return None

            # ENHANCED: Check time filters first
            valid_time, time_reason = self.time_filter_valid()
            if not valid_time:
                logger.debug(f"Skipping {symbol} - Invalid trading time: {time_reason}")
                return None

            token = symbol_info.get("token")
            prev_close = symbol_info.get("prev_close")

            if not token or not prev_close:
                logger.debug(f"Missing token or prev_close for {symbol}")
                return None

            # Fetch live price
            live_price = fetch_live_quote(self.kite_client, symbol)
            if live_price is None:
                logger.debug(f"Could not fetch live price for {symbol}")
                return None

            # Fetch minute data
            minute_df = fetch_minute_data(self.kite_client, token, minutes=30)  # ENHANCED: Increased from 20 to 30
            if minute_df is None or minute_df.empty:
                logger.debug(f"Could not fetch minute data for {symbol}")
                return None

            # Compute enhanced technical indicators
            enhanced_df, indicators = compute_enhanced_indicators(minute_df)

            # ENHANCED: Check volatility filter immediately
            volatility = indicators.get("volatility", 0)
            if volatility < MIN_VOLATILITY:
                logger.debug(f"Skipping {symbol} - Low volatility: {volatility:.4f}")
                return None

            # Generate enhanced rule-based signal
            rule_signal, rule_details = evaluate_enhanced_scalping_conditions(
                symbol, live_price, prev_close, indicators
            )

            if rule_signal:
                logger.signal(f"{symbol}: Enhanced rule-based signal - {rule_signal} "
                              f"[Bullish: {rule_details['bullish_score']}, Bearish: {rule_details['bearish_score']}]")

            # Generate ML signal
            ml_signal = None
            ml_prob = 0

            # Try to get ML prediction if model exists
            ml_result = self.ml_predictor.predict(clean_symbol, minute_df)
            if ml_result:
                ml_signal, ml_prob = ml_result

                # ENHANCED: Require higher ML confidence (80% -> 85%)
                if ml_signal and ml_prob > 0.85:
                    logger.signal(f"{symbol}: ML signal - {ml_signal} (high confidence: {ml_prob:.2f})")

            # ENHANCED: Enhanced signal combination logic
            trade_signal = None
            signal_quality = 0.0  # For position sizing

            if rule_signal is not None and ml_signal is not None and rule_signal == ml_signal:
                # Both signals agree - highest quality
                trade_signal = rule_signal
                signal_quality = min(1.0, 0.7 + ml_prob * 0.3)  # 0.7-1.0 based on ML confidence
                signal_source = "COMBINED"
            # ENHANCED: Only take ML signals with very high confidence and strong ADX
            elif ml_signal is not None and ml_prob > 0.9 and indicators.get("adx", 0) > ADX_STRONG:
                trade_signal = ml_signal
                signal_quality = ml_prob * 0.8  # Slightly lower than combined (max 0.8)
                signal_source = "HIGH-CONF ML"

            if trade_signal:
                logger.signal(f"{symbol}: ** {signal_source} SIGNAL - {trade_signal} (Quality: {signal_quality:.2f})")

                # Return enhanced signal information with quality score for position sizing
                return {
                    "symbol": symbol,
                    "clean_symbol": clean_symbol,
                    "signal": trade_signal,
                    "source": signal_source,
                    "live_price": live_price,
                    "prev_close": prev_close,
                    "price_change_pct": (live_price / prev_close - 1) * 100,
                    "volatility": volatility,
                    "adx": indicators.get("adx", 0),
                    "signal_quality": signal_quality,
                    "indicators": indicators
                }

            return None

        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            return None

    def scan_symbols(self, symbols_data, max_workers=5):
        """
        Scan multiple symbols and generate trading signals.

        Args:
            symbols_data: Dict with symbol info
            max_workers: Max number of parallel workers

        Returns:
            list: List of signal dictionaries
        """
        signals = []
        symbols_processed = 0

        if not symbols_data:
            logger.warning("No symbols data provided for scanning")
            return signals

        logger.info(f"Scanning {len(symbols_data)} symbols for trading signals")

        # Process symbols in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all symbols to the executor
            future_to_symbol = {
                executor.submit(self.process_symbol, symbol, info): symbol
                for symbol, info in symbols_data.items()
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                symbols_processed += 1

                # Log progress periodically
                if symbols_processed % 10 == 0:
                    logger.info(f"Processed {symbols_processed}/{len(symbols_data)} symbols")

                try:
                    signal = future.result()
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        logger.info(f"Completed scanning {symbols_processed} symbols. Found {len(signals)} signals.")

        # ENHANCED: Sort signals by signal quality and volatility
        if signals:
            signals.sort(key=lambda x: (x.get("signal_quality", 0), x.get("volatility", 0)), reverse=True)

        return signals

    def get_symbols_data(self, symbols):
        """
        Prepare symbols data for scanning.

        Args:
            symbols: List of symbols

        Returns:
            dict: Symbol info including tokens and prev_close values
        """
        symbols_data = {}

        logger.info(f"Preparing data for {len(symbols)} symbols")

        for raw_symbol in symbols:
            try:
                # Clean symbol for consistent processing
                clean_symbol = raw_symbol.strip().upper()
                if ":" in clean_symbol:
                    exchange, symbol = clean_symbol.split(":", 1)
                else:
                    symbol = clean_symbol

                # Get instrument token
                token = self.kite_client.get_instrument_token("NSE", symbol)
                if token is None:
                    logger.warning(f"Could not find instrument token for {symbol}. Skipping.")
                    continue

                # Get previous close price
                prev_close = fetch_previous_close(self.kite_client, token)
                if prev_close is None:
                    logger.warning(f"Could not fetch previous close for {symbol}. Skipping.")
                    continue

                # Store token and previous close price
                # Create a clean version without -EQ suffix for model lookup
                symbol_no_suffix = symbol.strip().upper()
                if symbol_no_suffix.endswith("-EQ"):
                    symbol_no_suffix = symbol_no_suffix[:-3]

                symbols_data[symbol_no_suffix] = {
                    "token": token,
                    "prev_close": prev_close
                }

            except Exception as e:
                logger.error(f"Error preparing data for {raw_symbol}: {e}")

        logger.info(f"Prepared data for {len(symbols_data)} valid symbols")
        return symbols_data