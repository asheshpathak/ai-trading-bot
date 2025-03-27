#!/usr/bin/env python3
"""
Enhanced Trading Bot Script

This script runs the main trading bot with improved features:
- Enhanced signal generation with trend confirmation
- Dynamic position sizing based on signal quality and volatility
- Time-of-day filtering to avoid volatile periods
- Robust GTT order placement with retries
- Improved risk management
"""

import os
import time
import argparse
import threading
import traceback
from datetime import datetime
import pytz

from config.settings import (
    SYMBOLS_DIR, PICKLES_DIR, LOG_DIR, SCAN_INTERVAL,
    MIN_MARGIN_REQUIRED, ORDER_TYPE, MIN_VOLATILITY
)
from config.credentials import Credentials
from api.client import KiteClient
from models.ml_predictor import MLPredictor
from trading.signals import SignalGenerator
from trading.options import OptionsFinder
from trading.orders import OrderManager
from trading.risk import RiskManager
from utils.logger import initialize_logging
from utils.symbols import read_symbols_from_file


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument("--symbols", "-s", help="Path to symbols file")
    parser.add_argument("--interval", "-i", type=int, default=SCAN_INTERVAL,
                        help=f"Scan interval in seconds (default: {SCAN_INTERVAL})")
    parser.add_argument("--parallel", "-p", type=int, default=5,
                        help="Number of parallel symbol processors (default: 5)")
    parser.add_argument("--min-margin", "-m", type=float, default=MIN_MARGIN_REQUIRED,
                        help=f"Minimum margin required (default: {MIN_MARGIN_REQUIRED})")
    parser.add_argument("--log-level", "-l", default="INFO",
                        help="Logging level (default: INFO)")
    parser.add_argument("--dryrun", "-d", action="store_true",
                        help="Dry run mode (no actual orders)")
    parser.add_argument("--volatility", "-v", type=float, default=MIN_VOLATILITY,
                        help=f"Minimum volatility threshold (default: {MIN_VOLATILITY})")
    return parser.parse_args()


def log_positions_thread(order_manager):
    """
    Thread function to periodically log position information.

    Args:
        order_manager: OrderManager instance
    """
    logger = initialize_logging(console_level="INFO")
    logger.info("Started positions monitoring thread")

    while True:
        try:
            order_manager.log_positions()
        except Exception as e:
            logger.error(f"Error in positions thread: {e}")

        # Sleep for 5 seconds
        time.sleep(5)


def is_market_open():
    """
    Check if the market is currently open.

    Returns:
        bool: True if market is open, False otherwise
    """
    india_tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(india_tz)

    # Check if it's a weekday (0=Monday, 4=Friday)
    if now.weekday() > 4:  # Saturday=5, Sunday=6
        return False

    # Check if it's between 9:15 AM and 3:30 PM IST
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()

    # Initialize logging
    logger = initialize_logging(console_level=args.log_level)
    logger.info("=" * 60)
    logger.info("STARTING ENHANCED TRADING BOT")
    logger.info(f"Mode: {'DRY RUN' if args.dryrun else 'LIVE TRADING'}")
    logger.info("=" * 60)

    # Load credentials and initialize API client
    credentials = Credentials()
    creds = credentials.get_credentials()
    kite_client = KiteClient(creds['api_key'], creds['api_secret'])
    kite = kite_client.initialize()

    logger.info("API client initialized successfully")

    # Initialize components
    ml_predictor = MLPredictor(PICKLES_DIR)
    signal_generator = SignalGenerator(kite_client, ml_predictor)
    options_finder = OptionsFinder(kite_client)
    order_manager = OrderManager(kite_client)
    risk_manager = RiskManager()

    # Load models
    models = ml_predictor.load_all_models()
    logger.info(f"Loaded {len(models)} ML models for prediction")

    # Load symbols
    symbols = read_symbols_from_file(args.symbols)
    if not symbols:
        logger.error("No symbols found in symbols file")
        return

    logger.info(f"Loaded {len(symbols)} symbols for trading")

    # Start positions monitoring thread
    positions_thread = threading.Thread(
        target=log_positions_thread,
        args=(order_manager,),
        daemon=True
    )
    positions_thread.start()

    # Initial margin check
    margin_info = order_manager.check_margin()
    if margin_info["is_sufficient"]:
        logger.info(f"Initial margin: Available INR {margin_info['available_margin']:,.2f}, "
                    f"Required minimum: INR {args.min_margin:,.2f}")
    else:
        logger.warning(f"Initial margin: Available INR {margin_info['available_margin']:,.2f} "
                       f"is below required minimum of INR {args.min_margin:,.2f}")

    # Get symbol data once (tokens and previous close)
    symbols_data = signal_generator.get_symbols_data(symbols)

    # Main trading loop
    scan_count = 0
    daily_reset_done = False

    while True:
        # Check if market is open
        if not is_market_open():
            logger.info("Market is closed. Waiting for next market day.")

            # Daily reset when market is closed
            if not daily_reset_done:
                logger.info("Performing daily reset")
                # Reset traded symbols for next day
                signal_generator.reset_traded_symbols()
                daily_reset_done = True

            # Sleep for 15 minutes before checking again
            time.sleep(900)
            continue
        else:
            # Market is open, reset the flag
            daily_reset_done = False

        scan_count += 1
        logger.info("\n" + "=" * 60)
        logger.info(f"STARTING SCAN CYCLE #{scan_count}")
        logger.info("=" * 60)

        # Check account balance before each cycle
        margin_info = order_manager.check_margin()
        if margin_info["is_sufficient"]:
            logger.info(f"Margin check: Available INR {margin_info['available_margin']:,.2f}, "
                        f"Required minimum: INR {args.min_margin:,.2f}")
        else:
            logger.warning(f"Low margin: INR {margin_info['available_margin']:,.2f} "
                           f"is below minimum requirement of INR {args.min_margin:,.2f}")

        # Check for completed orders and place GTT orders if needed
        order_manager.update_trades_status()

        # Check risk limits
        if not risk_manager.check_daily_loss_limit() or not risk_manager.check_weekly_loss_limit():
            logger.warning("Risk limits exceeded. Stopping trading for today.")
            # Sleep for an hour before checking again
            time.sleep(3600)
            continue

        # Scan for signals
        scan_start_time = time.time()
        signals = signal_generator.scan_symbols(symbols_data, max_workers=args.parallel)

        # Process signals
        if signals:
            logger.info(f"Found {len(signals)} trading signals")

            # Process each signal
            for signal in signals:
                symbol = signal["symbol"]
                signal_type = signal["signal"]
                live_price = signal["live_price"]
                signal_quality = signal.get("signal_quality", 0.5)  # Default to medium quality
                volatility = signal.get("volatility", args.volatility)

                logger.info(f"Processing signal for {symbol}: {signal_type} (Quality: {signal_quality:.2f})")

                try:
                    # Look up appropriate option
                    option_ts, lot_size = options_finder.get_option_tradingsymbol(
                        symbol, signal_type, live_price
                    )

                    if not option_ts or not lot_size:
                        logger.warning(f"Could not find suitable option for {symbol}. Skipping.")
                        continue

                    # Get option's live price
                    option_price = kite_client.quote(f"NFO:{option_ts}")
                    if not option_price:
                        logger.warning(f"Could not fetch price for option {option_ts}. Skipping.")
                        continue

                    option_price = option_price[f"NFO:{option_ts}"]["last_price"]

                    # Calculate cost and check margin
                    total_cost, margin_required = order_manager.estimate_option_cost(
                        option_price, lot_size
                    )

                    logger.info(f"Option cost: Total INR {total_cost:,.2f}, "
                                f"Margin required: INR {margin_required:,.2f}")

                    # ENHANCED: Check risk rules with signal quality and volatility
                    should_trade, reason = risk_manager.should_take_trade(
                        symbol, option_price, lot_size, margin_info["available_margin"],
                        signal_quality=signal_quality, volatility=volatility
                    )

                    if not should_trade:
                        logger.info(f"Risk check failed for {symbol}: {reason}")
                        continue

                    # ENHANCED: Calculate position size based on signal quality and volatility
                    quantity = risk_manager.calculate_dynamic_position_size(
                        option_price, lot_size, margin_info["available_margin"],
                        signal_quality=signal_quality, volatility=volatility
                    )

                    # Calculate target and stoploss
                    vol_factor = 1 + max(0, volatility - 0.005) * 10
                    target_percent = 0.10 * vol_factor  # Base target 10%
                    stoploss_percent = 0.04 * vol_factor  # Base stoploss 4%

                    target_price = option_price * (1 + target_percent)
                    stop_loss = option_price * (1 - stoploss_percent)

                    # Round to 2 decimal places
                    target_price = round(target_price, 2)
                    stop_loss = round(stop_loss, 2)

                    # Create trading plan
                    logger.trade(f"""
TRADING PLAN FOR {symbol} ({option_ts}):
  Signal: {signal_type} (Quality: {signal_quality:.2f})
  Entry: INR {option_price:.2f}
  Target: INR {target_price:.2f} ({target_percent * 100:.1f}%)
  Stoploss: INR {stop_loss:.2f} ({stoploss_percent * 100:.1f}%)
  Quantity: {quantity}
  Total Cost: INR {total_cost:,.2f}
  Volatility: {volatility:.4f}
  ADX: {signal.get('adx', 0):.1f}
""")

                    # Place order if not dry run
                    if not args.dryrun:
                        order_id = order_manager.place_order(
                            option_ts, quantity, option_price, target_price, stop_loss
                        )

                        if order_id:
                            # Add to tracked symbols
                            signal_generator.add_traded_symbol(symbol)

                            # Update risk manager with position
                            risk_manager.update_position(option_ts, {
                                "symbol": symbol,
                                "option": option_ts,
                                "quantity": quantity,
                                "entry_price": option_price,
                                "value": option_price * quantity,
                                "signal_quality": signal_quality,
                                "volatility": volatility
                            })

                            # Update available margin
                            margin_info["available_margin"] -= margin_required

                            logger.trade(f"ORDER PLACED for {symbol}. Order ID: {order_id}")
                        else:
                            logger.error(f"Failed to place order for {symbol}")
                    else:
                        logger.info(f"DRY RUN: Order would be placed for {symbol}")

                except Exception as e:
                    logger.error(f"Error processing signal for {symbol}: {e}")

        else:
            logger.info("No trading signals found in this scan")

        # Check for time-based exits
        for symbol, trade_info in list(order_manager.active_trades.items()):
            if trade_info['status'] == 'ACTIVE':
                should_exit, reason = risk_manager.should_exit_by_time(trade_info['timestamp'])
                if should_exit:
                    logger.info(f"Time-based exit for {symbol}: {reason}")

                    if not args.dryrun:
                        # Exit position
                        if order_manager.exit_position(symbol, reason):
                            # Update risk manager
                            risk_manager.remove_position(symbol)
                    else:
                        logger.info(f"DRY RUN: Would exit position for {symbol}")

        # Log scan duration
        scan_duration = time.time() - scan_start_time
        logger.info(f"Scan completed in {scan_duration:.2f} seconds")

        # Check for completed orders again before sleeping
        order_manager.update_trades_status()

        # Sleep before next scan
        logger.info(f"Waiting {args.interval} seconds before next scan.")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print(traceback.format_exc())