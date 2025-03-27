import os
import json
import logging
import pytz
from datetime import datetime, timedelta
from config.settings import (
    MAX_DAILY_LOSS, MAX_WEEKLY_LOSS, MAX_POSITION_SIZE_PCT,
    MAX_POSITIONS, MIN_VOLATILITY, IDEAL_VOLATILITY,
    AVOID_MARKET_CLOSE_MINS
)

# Initialize logger
logger = logging.getLogger("TradingBot")


class RiskManager:
    def __init__(self, config=None):
        """
        Initialize risk manager with optional configuration.

        Args:
            config: Dictionary with risk parameters (optional)
        """
        # Default config values
        self.config = {
            'max_daily_loss': MAX_DAILY_LOSS,
            'max_weekly_loss': MAX_WEEKLY_LOSS,
            'max_position_size_pct': MAX_POSITION_SIZE_PCT,
            'max_total_position_pct': 50,  # Max 50% total exposure
            'max_positions': MAX_POSITIONS,
            'time_based_exit_minutes': 60,  # Exit after 60 mins
            'market_close_buffer_minutes': AVOID_MARKET_CLOSE_MINS
        }

        # Override with custom config if provided
        if config:
            self.config.update(config)

        # Initialize P&L tracking
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.trades_today = 0
        self.positions = {}  # Current positions

        # Load historical P&L if available
        self._load_pnl_history()

    def _load_pnl_history(self):
        """Load P&L history from a file"""
        try:
            # Create the file path using the current date
            today = datetime.now().strftime("%Y-%m-%d")
            history_file = f"logs/pnl_history_{today}.json"

            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    self.daily_pnl = history.get('daily_pnl', 0)
                    self.weekly_pnl = history.get('weekly_pnl', 0)
                    self.trades_today = history.get('trades_today', 0)
                    logger.info(f"Loaded P&L history: Daily P&L: {self.daily_pnl}, Weekly P&L: {self.weekly_pnl}")
            else:
                logger.info("No P&L history found for today. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading P&L history: {e}")
            # Continue with default values

    def _save_pnl_history(self):
        """Save P&L history to a file"""
        try:
            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)

            # Create the file path using the current date
            today = datetime.now().strftime("%Y-%m-%d")
            history_file = f"logs/pnl_history_{today}.json"

            # Save current P&L state
            history = {
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': self.weekly_pnl,
                'trades_today': self.trades_today,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)

            logger.debug(f"Saved P&L history to {history_file}")
        except Exception as e:
            logger.error(f"Error saving P&L history: {e}")

    def check_daily_loss_limit(self):
        """
        Check if we've hit the daily loss limit.

        Returns:
            bool: True if within limit, False if limit exceeded
        """
        max_loss = self.config['max_daily_loss']
        if self.daily_pnl < -max_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl} < -{max_loss}")
            return False
        return True

    def check_weekly_loss_limit(self):
        """
        Check if we've hit the weekly loss limit.

        Returns:
            bool: True if within limit, False if limit exceeded
        """
        max_loss = self.config['max_weekly_loss']
        if self.weekly_pnl < -max_loss:
            logger.warning(f"Weekly loss limit reached: {self.weekly_pnl} < -{max_loss}")
            return False
        return True

    # ENHANCED: Dynamic position sizing based on signal quality and volatility
    def calculate_dynamic_position_size(self, option_price, lot_size, account_value, signal_quality=0.5,
                                        volatility=0.01):
        """
        Calculate position size dynamically based on signal quality and volatility.

        Args:
            option_price: Current price of the option
            lot_size: Lot size for the option
            account_value: Available account value
            signal_quality: Quality score of the signal (0.0 to 1.0)
            volatility: Current volatility measure

        Returns:
            int: Quantity to trade
        """
        # Basic calculation using percentage of account
        max_position_value = (account_value * self.config['max_position_size_pct'] / 100)

        # Adjust based on signal quality (reduce size for weaker signals)
        quality_factor = 0.3 + (0.7 * signal_quality)  # Min 30%, max 100% of position

        # Adjust based on volatility (increase size for optimal volatility)
        if volatility < MIN_VOLATILITY:
            volatility_factor = 0.5  # Reduce size for low volatility
        elif volatility > IDEAL_VOLATILITY * 2:
            volatility_factor = 0.7  # Reduce size for extremely high volatility
        else:
            # Scale up to maximum at ideal volatility
            volatility_factor = min(1.0, volatility / IDEAL_VOLATILITY)

        # Combined adjustment factor
        adjustment_factor = quality_factor * volatility_factor

        # Adjusted position value
        adjusted_position_value = max_position_value * adjustment_factor

        # Calculate how many lots we can buy
        cost_per_lot = option_price * lot_size
        max_lots = int(adjusted_position_value / cost_per_lot)

        # Ensure at least 1 lot, but no more than reasonable
        lots_to_trade = max(1, min(max_lots, 5))  # Cap at 5 lots for safety

        logger.info(f"Dynamic position sizing: Signal quality factor: {quality_factor:.2f}, "
                    f"Volatility factor: {volatility_factor:.2f}, Adjustment: {adjustment_factor:.2f}")
        logger.info(f"Position sizing: {max_lots} lots possible, using {lots_to_trade} lots")

        return lots_to_trade * lot_size  # Return total quantity

    def calculate_position_size(self, option_price, lot_size, account_value):
        """
        Original method for calculating position size.
        Provided for backward compatibility, uses the dynamic method internally.

        Args:
            option_price: Current price of the option
            lot_size: Lot size for the option
            account_value: Available account value

        Returns:
            int: Quantity to trade
        """
        # Call the enhanced dynamic method with default quality and volatility
        return self.calculate_dynamic_position_size(option_price, lot_size, account_value)

    def should_take_trade(self, symbol, option_price, lot_size, account_value, signal_quality=0.5, volatility=0.01):
        """
        Determine if a new trade should be taken based on risk rules.

        Args:
            symbol: Symbol to trade
            option_price: Current price of the option
            lot_size: Lot size for the option
            account_value: Available account value
            signal_quality: Quality of the signal (0.0 to 1.0)
            volatility: Current volatility measure

        Returns:
            tuple: (bool, str) - (should_trade, reason)
        """
        # Check loss limits
        if not self.check_daily_loss_limit():
            return False, "Daily loss limit reached"

        if not self.check_weekly_loss_limit():
            return False, "Weekly loss limit reached"

        # Check number of open positions
        if len(self.positions) >= self.config['max_positions']:
            return False, "Maximum number of positions reached"

        # Check total exposure
        current_exposure = sum(pos.get('value', 0) for pos in self.positions.values())
        new_position_value = option_price * lot_size
        total_exposure = current_exposure + new_position_value

        if total_exposure > (account_value * self.config['max_total_position_pct'] / 100):
            return False, "Maximum total exposure reached"

        # Check market closing time
        if self.is_near_market_close():
            return False, "Too close to market close for new positions"

        # ENHANCED: Check minimum quality and volatility
        if signal_quality < 0.4:  # Only take reasonable quality signals
            return False, f"Signal quality too low ({signal_quality:.2f})"

        if volatility < MIN_VOLATILITY:
            return False, f"Volatility too low ({volatility:.4f})"

        # ENHANCED: Check minimum price for option
        if option_price < 5:  # Minimum price threshold
            return False, f"Option price too low ({option_price})"

        # All checks passed
        return True, None

    def is_near_market_close(self):
        """
        Check if we're near market closing time.

        Returns:
            bool: True if near close, False otherwise
        """
        india_tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(india_tz)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # Check if we're within the buffer period before market close
        time_to_close = (market_close - now).total_seconds() / 60  # minutes
        return time_to_close <= self.config['market_close_buffer_minutes']

    def should_exit_by_time(self, entry_time):
        """
        Check if a position should be exited based on time.

        Args:
            entry_time: Entry time string or datetime

        Returns:
            tuple: (bool, str) - (should_exit, reason)
        """
        india_tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(india_tz)

        # Convert entry_time to datetime if it's a string
        if isinstance(entry_time, str):
            entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            entry_time = entry_time.replace(tzinfo=india_tz)

        # Check if position has been held longer than max time
        time_held = (now - entry_time).total_seconds() / 60  # minutes
        if time_held >= self.config['time_based_exit_minutes']:
            return True, "Maximum holding time reached"

        # Also check if we're near market close
        if self.is_near_market_close():
            return True, "Approaching market close"

        return False, None

    def update_pnl(self, trade_pnl):
        """
        Update daily and weekly P&L tracking.

        Args:
            trade_pnl: P&L from the trade
        """
        self.daily_pnl += trade_pnl
        self.weekly_pnl += trade_pnl
        self.trades_today += 1

        # Save updated P&L to persistent storage
        self._save_pnl_history()

        logger.info(f"Updated P&L: Trade P&L: {trade_pnl}, Daily P&L: {self.daily_pnl}, "
                    f"Weekly P&L: {self.weekly_pnl}, Trades today: {self.trades_today}")

    def update_position(self, symbol, position_data):
        """
        Update a position in the tracking dictionary.

        Args:
            symbol: Symbol for the position
            position_data: Position data dictionary
        """
        self.positions[symbol] = position_data

    def remove_position(self, symbol):
        """
        Remove a position from tracking.

        Args:
            symbol: Symbol to remove
        """
        if symbol in self.positions:
            del self.positions[symbol]

    def get_positions_summary(self):
        """
        Get a summary of current positions.

        Returns:
            dict: Position summary
        """
        return {
            'count': len(self.positions),
            'symbols': list(self.positions.keys()),
            'total_value': sum(pos.get('value', 0) for pos in self.positions.values()),
            'positions': self.positions
        }

    def get_pnl_summary(self):
        """
        Get a summary of P&L tracking.

        Returns:
            dict: P&L summary
        """
        return {
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'trades_today': self.trades_today
        }