import numpy as np
import pandas as pd
import logging
from config.settings import (
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    OVERSOLD_THRESHOLD, OVERBOUGHT_THRESHOLD,
    SHORT_EMA, LONG_EMA, ADX_THRESHOLD, ADX_STRONG,
    MIN_VOLATILITY
)

# Initialize logger
logger = logging.getLogger("TradingBot")


def compute_rsi(prices, period=14):
    """
    Compute Relative Strength Index.

    Args:
        prices: Series of prices
        period: RSI period

    Returns:
        RSI value (0-100) or None if not enough data
    """
    try:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-6)  # Add small constant to avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        if not rsi.empty:
            return rsi.iloc[-1]
        else:
            return None
    except Exception as e:
        logger.error(f"Error computing RSI: {e}")
        return None


# ENHANCED: New function to compute ADX
def compute_adx(df, period=14):
    """
    Compute Average Directional Index (ADX) for trend strength.

    Args:
        df: DataFrame with OHLC data
        period: Period for ADX calculation

    Returns:
        DataFrame with ADX values
    """
    try:
        # Make a copy of the dataframe
        df_copy = df.copy()

        # Calculate True Range (TR)
        df_copy['high_low'] = df_copy['high'] - df_copy['low']
        df_copy['high_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['low_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['tr'] = df_copy[['high_low', 'high_close', 'low_close']].max(axis=1)

        # Calculate Directional Movement (+DM and -DM)
        df_copy['+dm'] = np.where((df_copy['high'] - df_copy['high'].shift(1)) >
                                  (df_copy['low'].shift(1) - df_copy['low']),
                                  np.maximum(df_copy['high'] - df_copy['high'].shift(1), 0),
                                  0)
        df_copy['-dm'] = np.where((df_copy['low'].shift(1) - df_copy['low']) >
                                  (df_copy['high'] - df_copy['high'].shift(1)),
                                  np.maximum(df_copy['low'].shift(1) - df_copy['low'], 0),
                                  0)

        # Calculate Smoothed True Range and Directional Movement
        df_copy['tr_smoothed'] = df_copy['tr'].rolling(window=period).sum()
        df_copy['+dm_smoothed'] = df_copy['+dm'].rolling(window=period).sum()
        df_copy['-dm_smoothed'] = df_copy['-dm'].rolling(window=period).sum()

        # Calculate Directional Indicators (+DI and -DI)
        df_copy['+di'] = 100 * (df_copy['+dm_smoothed'] / df_copy['tr_smoothed'].replace(0, 1e-6))
        df_copy['-di'] = 100 * (df_copy['-dm_smoothed'] / df_copy['tr_smoothed'].replace(0, 1e-6))

        # Calculate DX (Directional Index)
        df_copy['dx'] = 100 * (abs(df_copy['+di'] - df_copy['-di']) /
                               (df_copy['+di'] + df_copy['-di']).replace(0, 1e-6))

        # Calculate ADX (smoothed DX)
        df_copy['adx'] = df_copy['dx'].rolling(window=period).mean()

        return df_copy
    except Exception as e:
        logger.error(f"Error computing ADX: {e}")
        return df


# ENHANCED: New function to compute VWAP bands
def compute_vwap_bands(df, stddev_multiplier=2.0):
    """
    Compute VWAP and Standard Deviation Bands.

    Args:
        df: DataFrame with OHLCV data
        stddev_multiplier: Multiplier for standard deviation bands

    Returns:
        DataFrame with VWAP and bands
    """
    try:
        # Make a copy of the dataframe
        df_copy = df.copy()

        # Calculate typical price
        df_copy["typical_price"] = (df_copy["high"] + df_copy["low"] + df_copy["close"]) / 3

        # Calculate VWAP
        df_copy["vwap"] = (df_copy["typical_price"] * df_copy["volume"]).cumsum() / df_copy["volume"].cumsum()

        # Calculate deviation from VWAP
        df_copy["deviation"] = df_copy["typical_price"] - df_copy["vwap"]

        # Calculate squared deviation
        df_copy["squared_deviation"] = df_copy["deviation"] ** 2

        # Calculate cumulative squared deviation
        df_copy["sum_squared_deviation"] = (df_copy["squared_deviation"] * df_copy["volume"]).cumsum()

        # Calculate standard deviation
        df_copy["stddev"] = np.sqrt(df_copy["sum_squared_deviation"] / df_copy["volume"].cumsum())

        # Calculate VWAP bands
        df_copy["vwap_upper"] = df_copy["vwap"] + (stddev_multiplier * df_copy["stddev"])
        df_copy["vwap_lower"] = df_copy["vwap"] - (stddev_multiplier * df_copy["stddev"])

        return df_copy
    except Exception as e:
        logger.error(f"Error computing VWAP bands: {e}")
        return df


def compute_enhanced_indicators(df):
    """
    Compute enhanced technical indicators for trading including:
    - Traditional indicators (RSI, MACD, Bollinger)
    - VWAP and VWAP bands
    - EMA crossovers
    - Price action patterns
    - ADX and trend strength

    Args:
        df: DataFrame with OHLCV data

    Returns:
        tuple: (DataFrame with indicators, dict of latest indicator values)
    """
    try:
        # Traditional indicators
        df["sma20"] = df["close"].rolling(window=20, min_periods=5).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False, min_periods=5).mean()
        df["ema9"] = df["close"].ewm(span=9, adjust=False, min_periods=3).mean()
        df["ema12"] = df["close"].ewm(span=12, adjust=False, min_periods=3).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False, min_periods=6).mean()

        # MACD
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=3).mean()

        # Bollinger Bands
        std20 = df["close"].rolling(window=20, min_periods=5).std()
        df["upper_band"] = df["sma20"] + 2 * std20
        df["lower_band"] = df["sma20"] - 2 * std20

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=4).mean()
        avg_loss = loss.rolling(window=14, min_periods=4).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Calculate VWAP (Volume Weighted Average Price)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["price_volume"] = df["typical_price"] * df["volume"]
        df["cumulative_price_volume"] = df["price_volume"].cumsum()
        df["cumulative_volume"] = df["volume"].cumsum()
        df["vwap"] = df["cumulative_price_volume"] / df["cumulative_volume"]

        # EMA crossover signals
        df["ema_crossover"] = (df["ema9"] > df["ema20"]) & (df["ema9"].shift(1) <= df["ema20"].shift(1))
        df["ema_crossunder"] = (df["ema9"] < df["ema20"]) & (df["ema9"].shift(1) >= df["ema20"].shift(1))

        # Price Action Patterns
        # Bullish Engulfing
        df["bullish_engulfing"] = (
                (df["close"] > df["open"]) &  # Current candle is bullish
                (df["open"].shift(1) > df["close"].shift(1)) &  # Previous candle was bearish
                (df["open"] < df["close"].shift(1)) &  # Current open below previous close
                (df["close"] > df["open"].shift(1))  # Current close above previous open
        )

        # Bearish Engulfing
        df["bearish_engulfing"] = (
                (df["open"] > df["close"]) &  # Current candle is bearish
                (df["close"].shift(1) > df["open"].shift(1)) &  # Previous candle was bullish
                (df["open"] > df["close"].shift(1)) &  # Current open above previous close
                (df["close"] < df["open"].shift(1))  # Current close below previous open
        )

        # Doji detection (body is small compared to range)
        body_size = abs(df["close"] - df["open"])
        candle_range = df["high"] - df["low"]
        df["doji"] = body_size < (0.1 * candle_range)  # Body is less than 10% of range

        # Hammer detection (small body at top, long lower wick)
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
        df["hammer"] = (
                (body_size < (0.3 * candle_range)) &  # Small body
                (lower_wick > (2 * body_size)) &  # Lower wick at least 2x body
                (upper_wick < (0.1 * candle_range))  # Small upper wick
        )

        # Returns and volatility
        df["returns"] = df["close"].pct_change()

        # Calculate volatility with adaptive window size
        data_points = len(df)
        if data_points >= 20:
            df["volatility"] = df["returns"].rolling(window=20, min_periods=5).std()
        elif data_points >= 10:
            df["volatility"] = df["returns"].rolling(window=10, min_periods=3).std()
        else:
            df["volatility"] = df["returns"].rolling(window=5, min_periods=2).std()

        # Fill any NaN values in volatility
        if df["volatility"].isna().any():
            # If there are still NaN values, use the mean volatility or a default value
            mean_volatility = df["volatility"].mean()
            if pd.isna(mean_volatility):
                # If all values are NaN, use a reasonable default
                df["volatility"] = 0.01  # 1% volatility as a conservative default
            else:
                # Fill NaN with the mean of non-NaN values
                df["volatility"] = df["volatility"].fillna(mean_volatility)

        # VWAP indicators
        df["above_vwap"] = df["close"] > df["vwap"]
        df["price_crossing_vwap"] = (df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))

        # Momentum indicators
        df["momentum"] = df["close"] - df["close"].shift(4)  # 4-period momentum
        df["momentum"] = df["momentum"].fillna(0)  # Fill NaN momentum values

        # ENHANCED: Add momentum acceleration (rate of change of momentum)
        df["momentum_acceleration"] = df["momentum"] - df["momentum"].shift(3)
        df["momentum_acceleration"] = df["momentum_acceleration"].fillna(0)

        # ENHANCED: Add volume trend indicators
        df["volume_ma"] = df["volume"].rolling(window=10, min_periods=3).mean()
        df["volume_increase"] = df["volume"] > df["volume_ma"] * 1.5  # 50% above average

        # ENHANCED: Add Rate of Change
        df["roc_3"] = df["close"].pct_change(3) * 100  # 3-period Rate of Change

        # ENHANCED: Add Bollinger Band squeeze detection (volatility contraction)
        df["bb_width"] = (df["upper_band"] - df["lower_band"]) / df["sma20"]
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(window=20, min_periods=5).mean()

        # ENHANCED: Add ADX calculation
        df = compute_adx(df)

        # ENHANCED: Add VWAP bands
        df = compute_vwap_bands(df)

        # Return only the latest values for efficiency
        latest_indicators = {
            # Traditional indicators
            "rsi": df["rsi"].iloc[-1] if not df.empty else None,
            "sma20": df["sma20"].iloc[-1] if not df.empty else None,
            "ema9": df["ema9"].iloc[-1] if not df.empty else None,
            "ema20": df["ema20"].iloc[-1] if not df.empty else None,
            "macd": df["macd"].iloc[-1] if not df.empty else None,
            "macd_signal": df["macd_signal"].iloc[-1] if not df.empty else None,
            "upper_band": df["upper_band"].iloc[-1] if not df.empty else None,
            "lower_band": df["lower_band"].iloc[-1] if not df.empty else None,

            # VWAP indicators
            "vwap": df["vwap"].iloc[-1] if not df.empty else None,
            "above_vwap": df["above_vwap"].iloc[-1] if not df.empty else None,
            "price_crossing_vwap": df["price_crossing_vwap"].iloc[-1] if not df.empty else None,

            # ENHANCED: VWAP bands
            "vwap_upper": df["vwap_upper"].iloc[-1] if not df.empty else None,
            "vwap_lower": df["vwap_lower"].iloc[-1] if not df.empty else None,

            # Pattern indicators
            "ema_crossover": df["ema_crossover"].iloc[-1] if not df.empty else None,
            "ema_crossunder": df["ema_crossunder"].iloc[-1] if not df.empty else None,
            "bullish_engulfing": df["bullish_engulfing"].iloc[-1] if not df.empty else None,
            "bearish_engulfing": df["bearish_engulfing"].iloc[-1] if not df.empty else None,
            "doji": df["doji"].iloc[-1] if not df.empty else None,
            "hammer": df["hammer"].iloc[-1] if not df.empty else None,

            # Volatility and momentum
            "volatility": df["volatility"].iloc[-1] if not df.empty else 0.01,  # Default value
            "momentum": df["momentum"].iloc[-1] if not df.empty else None,

            # ENHANCED: New indicators
            "adx": df["adx"].iloc[-1] if not df.empty else None,
            "plus_di": df["+di"].iloc[-1] if not df.empty else None,
            "minus_di": df["-di"].iloc[-1] if not df.empty else None,
            "volume_increase": df["volume_increase"].iloc[-1] if not df.empty else None,
            "roc_3": df["roc_3"].iloc[-1] if not df.empty else None,
            "bb_squeeze": df["bb_squeeze"].iloc[-1] if not df.empty else None,
            "momentum_acceleration": df["momentum_acceleration"].iloc[-1] if not df.empty else None,
        }

        return df, latest_indicators

    except Exception as e:
        logger.error(f"Error computing indicators: {e}")
        return df, {}


def prepare_features_for_prediction(minute_df):
    """
    Prepares features for ML model prediction, matching the features used in training.

    Args:
        minute_df: DataFrame with OHLCV data

    Returns:
        DataFrame with features ready for prediction
    """
    # Implementation remains largely the same from previous version
    # ...
    try:
        # Check if we have enough data points
        if len(minute_df) < 30:
            logger.warning(f"Limited data points ({len(minute_df)}) available for feature calculation")

        # Make a copy to avoid modifying the original
        df = minute_df.copy()

        # Calculate the same features used in training
        df["returns"] = df["close"].pct_change()

        # Use min_periods to allow calculation with fewer data points
        df["sma20"] = df["close"].rolling(window=20, min_periods=5).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False, min_periods=5).mean()
        df["ema12"] = df["close"].ewm(span=12, adjust=False, min_periods=3).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False, min_periods=6).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=3).mean()

        # Calculate RSI with smaller min_periods
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=4).mean()
        avg_loss = loss.rolling(window=14, min_periods=4).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Use forward fill then backward fill to handle NaN values
        df = df.ffill().bfill()

        # Check for NaN values after filling
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Still have NaN values after filling in columns: {nan_cols}")

            # Handle special case with short data window
            if len(df) < 10:
                logger.info("Very short data window, using simple imputation for missing values")

                # Calculate safe mean values
                close_mean = df['close'].mean() if not df['close'].isna().all() else 0

                # Fill missing data with reasonable values
                fill_values = {
                    'returns': 0,
                    'sma20': close_mean,
                    'ema20': close_mean,
                    'ema12': close_mean,
                    'ema26': close_mean,
                    'macd': 0,
                    'macd_signal': 0,
                    'rsi': 50
                }

                # Only fill columns that have NaN values
                for col in nan_cols:
                    if col in fill_values:
                        df[col] = df[col].fillna(fill_values[col])
                    else:
                        # For any other columns not explicitly handled
                        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
            else:
                # For longer data windows, do targeted column-by-column filling
                for col in nan_cols:
                    if col in ['returns', 'macd', 'macd_signal']:
                        df[col] = df[col].fillna(0)
                    elif col in ['sma20', 'ema20', 'ema12', 'ema26']:
                        df[col] = df[col].fillna(df['close'].mean())
                    elif col == 'rsi':
                        df[col] = df[col].fillna(50)
                    else:
                        # For any other numeric columns
                        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                            df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0)
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')

        # Make sure we have at least one row
        if len(df) == 0:
            logger.error("No data available after preparing features - dataset is empty")
            return None

        return df

    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return None


def prepare_features_for_training(df):
    """
    Prepares features for model training, including creating the target variable.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with features and target ready for training
    """
    try:
        # Calculate technical indicators
        df["returns"] = df["close"].pct_change()
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ENHANCED: Add ADX
        df = compute_adx(df)

        # ENHANCED: Add momentum features
        df["momentum"] = df["close"] - df["close"].shift(4)  # 4-period momentum
        df["momentum_acceleration"] = df["momentum"] - df["momentum"].shift(3)

        # ENHANCED: Add volatility feature
        df["volatility"] = df["returns"].rolling(window=20).std()

        # Drop NaN values created by indicator calculations
        df.dropna(inplace=True)

        # Create target variable - 1 if next close is higher, 0 if lower or equal
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        # Drop last row (NaN target)
        df.dropna(inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error preparing features for training: {e}")
        return None


def evaluate_enhanced_scalping_conditions(symbol, live_price, prev_close, indicators):
    """
    Evaluates enhanced scalping conditions for both bullish and bearish signals.
    Includes stronger filters and trend alignment checks.

    Args:
        symbol: Symbol being evaluated
        live_price: Current price
        prev_close: Previous close price
        indicators: Dictionary of indicators

    Returns:
        tuple: (signal, details)
    """
    # Extract indicators for readability
    rsi = indicators.get("rsi")
    adx = indicators.get("adx")
    plus_di = indicators.get("plus_di")
    minus_di = indicators.get("minus_di")
    volatility = indicators.get("volatility", 0)
    vwap = indicators.get("vwap")
    vwap_upper = indicators.get("vwap_upper")
    vwap_lower = indicators.get("vwap_lower")

    # Check for minimum volatility
    if volatility < MIN_VOLATILITY:
        return None, {
            "symbol": symbol,
            "reason": f"Insufficient volatility: {volatility:.4f} < {MIN_VOLATILITY:.4f}"
        }

    # Price change from previous close
    price_change_pct = (live_price / prev_close - 1) * 100

    # Enhanced Bullish Conditions (Call Options)
    bullish_conditions = {
        # Strong reversal conditions
        "price_below_threshold": live_price <= prev_close * OVERSOLD_THRESHOLD,
        "rsi_oversold": rsi < RSI_OVERSOLD if rsi else False,
        "price_below_vwap": live_price < vwap if vwap else False,

        # Momentum confirmation
        "adx_strong_enough": adx > ADX_THRESHOLD if adx else False,
        "momentum_positive": indicators.get("momentum", 0) > 0,
        "volume_spike": indicators.get("volume_increase", False),

        # Pattern confirmation
        "bullish_engulfing": indicators.get("bullish_engulfing", False),
        "hammer_pattern": indicators.get("hammer", False),
        "macd_bullish": indicators.get("macd", 0) > indicators.get("macd_signal", 0),
        "ema_crossover": indicators.get("ema_crossover", False),

        # Trend alignment (new)
        "bullish_di": plus_di > minus_di if plus_di and minus_di else False,
        "vwap_support": live_price > vwap_lower if vwap_lower else False,
        "acceleration_positive": indicators.get("momentum_acceleration", 0) > 0
    }

    # Enhanced Bearish Conditions (Put Options)
    bearish_conditions = {
        # Strong reversal conditions
        "price_above_threshold": live_price >= prev_close * OVERBOUGHT_THRESHOLD,
        "rsi_overbought": rsi > RSI_OVERBOUGHT if rsi else False,
        "price_above_vwap": live_price > vwap if vwap else False,

        # Momentum confirmation
        "adx_strong_enough": adx > ADX_THRESHOLD if adx else False,
        "momentum_negative": indicators.get("momentum", 0) < 0,
        "volume_spike": indicators.get("volume_increase", False),

        # Pattern confirmation
        "bearish_engulfing": indicators.get("bearish_engulfing", False),
        "doji_at_top": indicators.get("doji", False) and live_price > indicators.get("ema20", 0),
        "macd_bearish": indicators.get("macd", 0) < indicators.get("macd_signal", 0),
        "ema_crossunder": indicators.get("ema_crossunder", False),

        # Trend alignment (new)
        "bearish_di": minus_di > plus_di if plus_di and minus_di else False,
        "vwap_resistance": live_price < vwap_upper if vwap_upper else False,
        "acceleration_negative": indicators.get("momentum_acceleration", 0) < 0
    }

    # Define critical conditions with higher standards
    # Now require more conditions and include trend alignment checks
    bullish_core = sum([
        bullish_conditions["price_below_threshold"],
        bullish_conditions["rsi_oversold"],
        bullish_conditions["adx_strong_enough"],
        bullish_conditions["bullish_di"]
    ])

    bullish_total = sum(bullish_conditions.values())

    bearish_core = sum([
        bearish_conditions["price_above_threshold"],
        bearish_conditions["rsi_overbought"],
        bearish_conditions["adx_strong_enough"],
        bearish_conditions["bearish_di"]
    ])

    bearish_total = sum(bearish_conditions.values())

    # Determine signal with enhanced requirements
    # Now require at least 3 core conditions and 7 total conditions
    rule_signal = None
    if bullish_total >= 7 and bullish_core >= 3:
        rule_signal = "Buy Call Option"
    elif bearish_total >= 7 and bearish_core >= 3:
        rule_signal = "Buy Put Option"

    # Prepare detailed feedback
    details = {
        "symbol": symbol,
        "live_price": live_price,
        "prev_close": prev_close,
        "price_change_pct": price_change_pct,
        "volatility": volatility,
        "adx": adx,
        "indicators": indicators,
        "bullish_conditions": bullish_conditions,
        "bearish_conditions": bearish_conditions,
        "bullish_score": f"{bullish_total}/13 (core: {bullish_core}/4)",
        "bearish_score": f"{bearish_total}/13 (core: {bearish_core}/4)",
        "rule_signal": rule_signal
    }

    return rule_signal, details


def get_atm_strike(price, strike_interval=50):
    """
    Calculates the at-the-money strike price.

    Args:
        price: Current price
        strike_interval: Strike price interval

    Returns:
        int: ATM strike price
    """
    return round(price / strike_interval) * strike_interval