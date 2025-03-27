import os
import logging
import logging.handlers
from datetime import datetime


class TradingLogger:
    """
    Custom logger for trading bot with specialized log levels and formatting.
    """
    # Custom log levels
    SIGNAL = 25  # Between INFO and WARNING
    TRADE = 35  # Between WARNING and ERROR

    def __init__(self, log_dir="logs", log_level="INFO", file_log_level="DEBUG",
                 console_format="standard", file_rotation=True):
        """
        Initialize logger with customizable settings.

        Args:
            log_dir: Directory to store log files
            log_level: Console logging level (default "INFO")
            file_log_level: File logging level (default "DEBUG")
            console_format: Format for console logs ("minimal", "standard", or "detailed")
            file_rotation: Whether to use rotating file handler
        """
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Convert string log levels to constants
        console_level = self._get_log_level(log_level)
        file_level = self._get_log_level(file_log_level)

        # Create logger
        self.logger = logging.getLogger("TradingBot")
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        self.logger.propagate = False  # Don't propagate to root logger

        # Register custom log levels
        logging.addLevelName(self.SIGNAL, "SIGNAL")
        logging.addLevelName(self.TRADE, "TRADE")

        # Add custom log methods
        logging.Logger.signal = self._signal_method
        logging.Logger.trade = self._trade_method

        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Configure console handler
        self._setup_console_handler(console_level, console_format)

        # Configure file handlers
        today = datetime.now().strftime("%Y-%m-%d")

        # Main log file with all messages
        main_log_file = os.path.join(log_dir, f"trading_bot_{today}.log")
        self._setup_file_handler(main_log_file, file_level, "detailed", file_rotation)

        # Separate trade log file for just trade execution info
        trade_log_file = os.path.join(log_dir, f"trades_{today}.log")
        self._setup_trade_file_handler(trade_log_file, "detailed", file_rotation)

        # Signal log file for trading signals
        signal_log_file = os.path.join(log_dir, f"signals_{today}.log")
        self._setup_signal_file_handler(signal_log_file, "detailed", file_rotation)

    def _get_log_level(self, level_str):
        """Convert string log level to logging constant"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "SIGNAL": self.SIGNAL,
            "TRADE": self.TRADE
        }
        return level_map.get(level_str.upper(), logging.INFO)

    def _get_formatter(self, format_type):
        """Get formatter based on format type"""
        if format_type == "minimal":
            return logging.Formatter('%(levelname)s: %(message)s')
        elif format_type == "standard":
            return logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                     datefmt='%H:%M:%S')
        elif format_type == "detailed":
            return logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        else:
            return logging.Formatter('%(levelname)s: %(message)s')

    def _setup_console_handler(self, level, format_type):
        """Set up console handler with specified level and format"""
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(self._get_formatter(format_type))
        self.logger.addHandler(console)

    def _setup_file_handler(self, filename, level, format_type, use_rotation=True):
        """Set up file handler with optional rotation"""
        if use_rotation:
            handler = logging.handlers.RotatingFileHandler(
                filename, maxBytes=10 * 1024 * 1024, backupCount=5)
        else:
            handler = logging.FileHandler(filename)

        handler.setLevel(level)
        handler.setFormatter(self._get_formatter(format_type))
        self.logger.addHandler(handler)

    def _setup_trade_file_handler(self, filename, format_type, use_rotation=True):
        """Set up specialized file handler for trade logs only"""
        if use_rotation:
            handler = logging.handlers.RotatingFileHandler(
                filename, maxBytes=5 * 1024 * 1024, backupCount=5)
        else:
            handler = logging.FileHandler(filename)

        handler.setLevel(self.TRADE)
        handler.setFormatter(self._get_formatter(format_type))

        # Add a filter to only include TRADE level logs
        handler.addFilter(lambda record: record.levelno == self.TRADE)

        self.logger.addHandler(handler)

    def _setup_signal_file_handler(self, filename, format_type, use_rotation=True):
        """Set up specialized file handler for signal logs only"""
        if use_rotation:
            handler = logging.handlers.RotatingFileHandler(
                filename, maxBytes=5 * 1024 * 1024, backupCount=5)
        else:
            handler = logging.FileHandler(filename)

        handler.setLevel(self.SIGNAL)
        handler.setFormatter(self._get_formatter(format_type))

        # Add a filter to only include SIGNAL level logs
        handler.addFilter(lambda record: record.levelno == self.SIGNAL)

        self.logger.addHandler(handler)

    def _signal_method(self, message, *args, **kwargs):
        """Custom signal logging method"""
        if self.isEnabledFor(TradingLogger.SIGNAL):
            formatted_message = f"SIGNAL: {message}"
            self._log(TradingLogger.SIGNAL, formatted_message, args=(), **kwargs)

    def _trade_method(self, message, *args, **kwargs):
        """Custom trade logging method - more condensed for console readability"""
        if self.isEnabledFor(TradingLogger.TRADE):
            # More compact format for TRADE logs
            formatted_message = f"TRADE: {message}"
            self._log(TradingLogger.TRADE, formatted_message, args=(), **kwargs)

    def get_logger(self):
        """Return the configured logger"""
        return self.logger


def initialize_logging(log_dir="logs", console_level="INFO", file_level="DEBUG", console_format="standard"):
    """
    Initialize and return a configured logger.

    Args:
        log_dir: Directory for log files
        console_level: Logging level for console output
        file_level: Logging level for file output
        console_format: Format for console logs

    Returns:
        logging.Logger: Configured logger
    """
    logger_manager = TradingLogger(
        log_dir=log_dir,
        log_level=console_level,
        file_log_level=file_level,
        console_format=console_format,
        file_rotation=True
    )
    return logger_manager.get_logger()