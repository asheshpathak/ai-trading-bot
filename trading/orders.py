import logging
import time
import traceback
from datetime import datetime
from config.settings import OPTION_EXCHANGE, ORDER_TYPE, BASE_TARGET_PERCENT, BASE_STOPLOSS_PERCENT

# Initialize logger
logger = logging.getLogger("TradingBot")


class OrderManager:
    def __init__(self, kite_client):
        """
        Initialize with KiteClient instance.

        Args:
            kite_client: KiteClient instance
        """
        self.kite_client = kite_client
        self.active_trades = {}

    def estimate_option_cost(self, option_price, lot_size):
        """
        Estimates the total cost and margin required for an option trade.

        Args:
            option_price: Current price of the option
            lot_size: Lot size for the option

        Returns:
            tuple: (total_cost, margin_with_buffer)
        """
        # Calculate the total cost of the option
        total_cost = option_price * lot_size

        # Estimate margin requirement (usually the full premium for long options)
        margin_required = total_cost

        # For additional safety, add some buffer for fees, etc.
        margin_with_buffer = margin_required * 1.05  # 5% buffer

        return total_cost, margin_with_buffer

    def check_margin(self):
        """
        Checks the available margin and returns detailed margin information.

        Returns:
            dict: Margin information
        """
        try:
            margins = self.kite_client.margins()
            equity = margins["equity"]

            available_cash = equity["available"]["cash"]
            used_margin = equity["utilised"]["exposure"] + equity["utilised"]["span"]
            available_margin = available_cash - used_margin

            margin_info = {
                "available_cash": available_cash,
                "used_margin": used_margin,
                "available_margin": available_margin,
                "is_sufficient": available_margin >= 0  # Min required margin is configurable elsewhere
            }

            return margin_info
        except Exception as e:
            logger.error(f"Error checking margins: {e}")
            return {
                "available_cash": 0,
                "used_margin": 0,
                "available_margin": 0,
                "is_sufficient": False,
                "error": str(e)
            }

    def place_order(self, option_tradingsymbol, quantity, entry, target, stoploss):
        """
        Place an order with proper handling of Zerodha's requirements.

        Args:
            option_tradingsymbol: Trading symbol for the option
            quantity: Number of options to buy
            entry: Entry price
            target: Target price
            stoploss: Stoploss price

        Returns:
            str: Order ID if successful, None otherwise
        """
        try:
            logger.info(f"Attempting to place order for {option_tradingsymbol}")
            logger.info(f"Quantity: {quantity}, Entry: {entry}, Target: {target}, Stoploss: {stoploss}")

            # For options, use limit orders with a buffer for better fill probability
            limit_price = entry * 1.05  # 5% above current price
            limit_price = round(limit_price, 2)  # Round to 2 decimal places

            logger.info(f"Using LIMIT order with price: {limit_price}")

            # Use NRML product type for overnight positions
            product_type = ORDER_TYPE
            logger.info(f"Using {product_type} product type for {option_tradingsymbol}")

            # Try regular limit order
            try:
                # Regular order parameters
                order_params = {
                    "exchange": OPTION_EXCHANGE,
                    "tradingsymbol": option_tradingsymbol,
                    "transaction_type": "BUY",
                    "quantity": quantity,
                    "product": product_type,
                    "order_type": "LIMIT",
                    "price": limit_price,
                    "validity": "DAY"
                }

                # Place the entry order
                order_id = self.kite_client.place_order(variety="regular", **order_params)
                logger.trade(f"LIMIT order placed for {option_tradingsymbol}. Order ID: {order_id}")

                # Store trade info for GTT placement after order is filled
                self.active_trades[option_tradingsymbol] = {
                    "underlying": option_tradingsymbol.split("25")[
                        0] if "25" in option_tradingsymbol else option_tradingsymbol,
                    "status": "PENDING",
                    "needs_sl_orders": True,
                    "entry_order_id": order_id,
                    "quantity": quantity,
                    "entry_price": entry,
                    "target_price": target,
                    "stoploss_price": stoploss,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "product_type": product_type
                }

                logger.info(
                    f"Added {option_tradingsymbol} to active_trades with PENDING status. GTT orders will be placed when filled.")
                return order_id

            except Exception as e1:
                if "compulsory physical delivery" in str(e1):
                    logger.warning(f"Order blocked due to compulsory physical delivery policy: {e1}")
                    return None

                logger.error(f"Error placing LIMIT order: {e1}")

                # If limit order fails, try a slightly higher price
                higher_price = limit_price * 1.05
                higher_price = round(higher_price, 2)
                logger.warning(f"Retrying with higher limit price: {higher_price}")

                order_params["price"] = higher_price

                try:
                    order_id = self.kite_client.place_order(variety="regular", **order_params)
                    logger.trade(
                        f"LIMIT order placed with higher price for {option_tradingsymbol}. Order ID: {order_id}")

                    # Store trade info for GTT placement after order is filled
                    self.active_trades[option_tradingsymbol] = {
                        "underlying": option_tradingsymbol.split("25")[
                            0] if "25" in option_tradingsymbol else option_tradingsymbol,
                        "status": "PENDING",
                        "needs_sl_orders": True,
                        "entry_order_id": order_id,
                        "quantity": quantity,
                        "entry_price": entry,
                        "target_price": target,
                        "stoploss_price": stoploss,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "product_type": product_type
                    }

                    logger.info(
                        f"Added {option_tradingsymbol} to active_trades with PENDING status. GTT orders will be placed when filled.")
                    return order_id
                except Exception as e2:
                    logger.error(f"Error placing LIMIT order with higher price: {e2}")
                    return None

        except Exception as e:
            logger.error(f"Error placing order for {option_tradingsymbol}: {e}")
            return None

    def place_gtt_orders(self, option_tradingsymbol, trade_info):
        """
        Places GTT orders for stop-loss and target for an option position.

        Args:
            option_tradingsymbol: Trading symbol for the option
            trade_info: Dictionary containing trade information

        Returns:
            tuple: (stoploss_gtt_id, target_gtt_id)
        """
        quantity = trade_info.get("quantity", 0)
        entry_price = trade_info.get("entry_price", 0)
        stoploss_price = trade_info.get("stoploss_price")
        target_price = trade_info.get("target_price")
        product_type = trade_info.get("product_type", ORDER_TYPE)

        stoploss_gtt_id = None
        target_gtt_id = None

        logger.info(f"Placing GTT orders for {option_tradingsymbol}")
        logger.info(f"Stop-loss: {stoploss_price}, Target: {target_price}, Quantity: {quantity}")

        try:
            # For all option buy orders (both calls and puts), stoploss is below entry price and target is above
            # Stoploss GTT - Triggers when price falls below stoploss_price
            sl_gtt_params = {
                "tradingsymbol": option_tradingsymbol,
                "exchange": OPTION_EXCHANGE,
                "trigger_type": "LTP",
                "trigger_values": [stoploss_price],
                "last_price": entry_price,
                "orders": [{
                    "transaction_type": "SELL",
                    "quantity": quantity,
                    "order_type": "MARKET",
                    "product": product_type
                }]
            }

            try:
                # Place the stoploss GTT
                sl_gtt_response = self.kite_client.place_gtt(**sl_gtt_params)
                stoploss_gtt_id = sl_gtt_response.get("trigger_id")
                logger.trade(
                    f"Stop-loss GTT placed for {option_tradingsymbol} at {stoploss_price}. Trigger ID: {stoploss_gtt_id}")
            except Exception as e:
                logger.error(f"Error placing stop-loss GTT: {e}")

                # If GTT fails, try placing a regular stop-loss order
                try:
                    sl_order_params = {
                        "exchange": OPTION_EXCHANGE,
                        "tradingsymbol": option_tradingsymbol,
                        "transaction_type": "SELL",
                        "quantity": quantity,
                        "product": product_type,
                        "order_type": "SL-M",
                        "trigger_price": stoploss_price,
                        "validity": "DAY"
                    }

                    sl_order_id = self.kite_client.place_order(variety="regular", **sl_order_params)
                    logger.trade(f"Fallback stop-loss order placed for {option_tradingsymbol}. Order ID: {sl_order_id}")
                    stoploss_gtt_id = "order:" + str(sl_order_id)  # Mark it as a regular order
                except Exception as e2:
                    logger.error(f"Error placing fallback stop-loss: {e2}")

            # Target GTT - Triggers when price rises above target_price
            tp_gtt_params = {
                "tradingsymbol": option_tradingsymbol,
                "exchange": OPTION_EXCHANGE,
                "trigger_type": "LTP",
                "trigger_values": [target_price],
                "last_price": entry_price,
                "orders": [{
                    "transaction_type": "SELL",
                    "quantity": quantity,
                    "order_type": "MARKET",
                    "product": product_type
                }]
            }

            try:
                # Place the target GTT
                tp_gtt_response = self.kite_client.place_gtt(**tp_gtt_params)
                target_gtt_id = tp_gtt_response.get("trigger_id")
                logger.trade(
                    f"Target GTT placed for {option_tradingsymbol} at {target_price}. Trigger ID: {target_gtt_id}")
            except Exception as e:
                logger.error(f"Error placing target GTT: {e}")

                # If GTT fails, try placing a regular limit order
                try:
                    tp_order_params = {
                        "exchange": OPTION_EXCHANGE,
                        "tradingsymbol": option_tradingsymbol,
                        "transaction_type": "SELL",
                        "quantity": quantity,
                        "product": product_type,
                        "order_type": "LIMIT",
                        "price": target_price,
                        "validity": "DAY"
                    }

                    tp_order_id = self.kite_client.place_order(variety="regular", **tp_order_params)
                    logger.trade(f"Fallback target order placed for {option_tradingsymbol}. Order ID: {tp_order_id}")
                    target_gtt_id = "order:" + str(tp_order_id)  # Mark it as a regular order
                except Exception as e2:
                    logger.error(f"Error placing fallback target order: {e2}")

            return stoploss_gtt_id, target_gtt_id

        except Exception as e:
            logger.error(f"Error setting up GTT orders: {e}")
            return None, None

    def update_trades_status(self):
        """
        Checks the status of all entry orders and places GTT orders when executed.
        Enhanced with retry mechanism for GTT order placement.
        This function should be called regularly to monitor order status.
        """
        if not self.active_trades:
            return

        try:
            # Get all orders to check their status
            orders = self.kite_client.orders()

            # First find all completed entry orders that need SL/TP orders
            for symbol, trade_info in list(self.active_trades.items()):
                if trade_info.get("status") == "PENDING" and trade_info.get("needs_sl_orders", False):
                    entry_order_id = trade_info.get("entry_order_id")

                    # Find the order in the list of orders
                    matching_orders = [o for o in orders if o.get("order_id") == entry_order_id]

                    if matching_orders:
                        order = matching_orders[0]
                        status = order.get("status")

                        # If the order is complete, place GTT orders
                        if status == "COMPLETE":
                            logger.info(f"Entry order {entry_order_id} for {symbol} is complete. Placing GTT orders.")

                            # Get the actual filled price from the order (might be different from our estimated entry)
                            filled_price = order.get("average_price") or trade_info.get("entry_price")

                            # Update trade info with actual price
                            trade_info["entry_price"] = filled_price

                            # Recalculate target and stoploss based on actual entry price
                            vol_factor = 1 + max(0, trade_info.get("volatility", 0.01) - 0.005) * 10
                            target_percent = BASE_TARGET_PERCENT * vol_factor
                            stoploss_percent = BASE_STOPLOSS_PERCENT * vol_factor

                            # For both calls and puts, target is above entry and stoploss is below
                            target_price = filled_price * (1 + target_percent)
                            stoploss_price = filled_price * (1 - stoploss_percent)

                            # Round to 2 decimal places
                            trade_info["target_price"] = round(target_price, 2)
                            trade_info["stoploss_price"] = round(stoploss_price, 2)

                            logger.info(
                                f"Updated target to {trade_info['target_price']} and stoploss to {trade_info['stoploss_price']}")

                            # ENHANCED: Add retry mechanism for GTT placement
                            max_retries = 3
                            gtt_success = False

                            for attempt in range(max_retries):
                                try:
                                    # Place GTT orders for SL and TP
                                    sl_gtt_id, tp_gtt_id = self.place_gtt_orders(symbol, trade_info)

                                    # Check if both GTT orders were placed successfully
                                    if sl_gtt_id and tp_gtt_id:
                                        # Update trade info
                                        trade_info["status"] = "ACTIVE"
                                        trade_info["needs_sl_orders"] = False
                                        trade_info["sl_gtt_id"] = sl_gtt_id
                                        trade_info["tp_gtt_id"] = tp_gtt_id
                                        self.active_trades[symbol] = trade_info

                                        logger.info(f"Updated trade status for {symbol} to ACTIVE with GTT orders")
                                        gtt_success = True
                                        break
                                    else:
                                        logger.warning(
                                            f"Failed to place both GTT orders on attempt {attempt + 1}, retrying...")
                                        time.sleep(1)  # Wait before retry
                                except Exception as e:
                                    logger.error(f"Error placing GTT orders on attempt {attempt + 1}: {e}")
                                    time.sleep(1)  # Wait before retry

                            # ENHANCED: If still failed after retries, emergency exit via market order
                            if not gtt_success:
                                logger.error(
                                    f"Failed to place GTT orders after {max_retries} attempts. Emergency exit for {symbol}")
                                self.exit_position(symbol, "Emergency exit - GTT placement failed")
                                continue

                        # If the order is rejected or cancelled, remove from active trades
                        elif status in ["REJECTED", "CANCELLED"]:
                            logger.info(
                                f"Entry order {entry_order_id} for {symbol} was {status}. Removing from active trades.")
                            self.active_trades.pop(symbol, None)

            # Check for triggered GTT orders or executed regular orders
            try:
                # Get all GTT orders
                gtt_orders = self.kite_client.get_gtts()

                # Check each active trade to see if GTTs have been triggered
                for symbol, trade_info in list(self.active_trades.items()):
                    if trade_info.get("status") == "ACTIVE":
                        sl_gtt_id = trade_info.get("sl_gtt_id")
                        tp_gtt_id = trade_info.get("tp_gtt_id")

                        # Skip if neither GTT ID is set
                        if not sl_gtt_id and not tp_gtt_id:
                            continue

                        sl_triggered = False
                        tp_triggered = False

                        # Check if SL GTT was triggered (if it's no longer in the list of active GTTs)
                        if sl_gtt_id and not sl_gtt_id.startswith("order:"):
                            sl_gtt_active = any(gtt.get("id") == int(sl_gtt_id) for gtt in gtt_orders)
                            if not sl_gtt_active:
                                sl_triggered = True
                                logger.trade(f"Stop-loss GTT triggered for {symbol}")

                        # Check if TP GTT was triggered
                        if tp_gtt_id and not tp_gtt_id.startswith("order:"):
                            tp_gtt_active = any(gtt.get("id") == int(tp_gtt_id) for gtt in gtt_orders)
                            if not tp_gtt_active:
                                tp_triggered = True
                                logger.trade(f"Target GTT triggered for {symbol}")

                        # For regular orders (not GTTs), check their status
                        if sl_gtt_id and sl_gtt_id.startswith("order:"):
                            order_id = sl_gtt_id.split(":")[1]
                            sl_orders = [o for o in orders if o.get("order_id") == order_id]
                            if sl_orders and sl_orders[0].get("status") == "COMPLETE":
                                sl_triggered = True
                                logger.trade(f"Stop-loss order executed for {symbol}")

                        if tp_gtt_id and tp_gtt_id.startswith("order:"):
                            order_id = tp_gtt_id.split(":")[1]
                            tp_orders = [o for o in orders if o.get("order_id") == order_id]
                            if tp_orders and tp_orders[0].get("status") == "COMPLETE":
                                tp_triggered = True
                                logger.trade(f"Target order executed for {symbol}")

                        # If either SL or TP was triggered, cancel the other and remove from active trades
                        if sl_triggered or tp_triggered:
                            # Cancel the other GTT/order
                            try:
                                if sl_triggered and tp_gtt_id:
                                    if tp_gtt_id.startswith("order:"):
                                        # Cancel regular order
                                        order_id = tp_gtt_id.split(":")[1]
                                        self.kite_client.cancel_order(variety="regular", order_id=order_id)
                                        logger.info(f"Cancelled target order {order_id} for {symbol}")
                                    else:
                                        # Delete GTT
                                        self.kite_client.delete_gtt(trigger_id=int(tp_gtt_id))
                                        logger.info(f"Cancelled target GTT {tp_gtt_id} for {symbol}")

                                if tp_triggered and sl_gtt_id:
                                    if sl_gtt_id.startswith("order:"):
                                        # Cancel regular order
                                        order_id = sl_gtt_id.split(":")[1]
                                        self.kite_client.cancel_order(variety="regular", order_id=order_id)
                                        logger.info(f"Cancelled stop-loss order {order_id} for {symbol}")
                                    else:
                                        # Delete GTT
                                        self.kite_client.delete_gtt(trigger_id=int(sl_gtt_id))
                                        logger.info(f"Cancelled stop-loss GTT {sl_gtt_id} for {symbol}")
                            except Exception as e:
                                logger.error(f"Error cancelling GTT/order: {e}")

                            # Update the position with exit info
                            exit_type = "Stop-Loss" if sl_triggered else "Target"
                            exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Save trade information before removing
                            trade_info["exit_type"] = exit_type
                            trade_info["exit_time"] = exit_time

                            # Store completed trade (for history/analysis) before removing
                            completed_trade = self.active_trades.pop(symbol, None)
                            logger.info(f"Trade for {symbol} completed via {exit_type}. Removed from active trades.")

            except Exception as e:
                logger.error(f"Error checking GTT status: {e}")
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            logger.error(traceback.format_exc())

    def exit_position(self, symbol, reason="Manual exit"):
        """
        Exit a specific position immediately.

        Args:
            symbol: Symbol to exit
            reason: Reason for exiting

        Returns:
            bool: True if successful, False otherwise
        """
        if symbol not in self.active_trades:
            logger.warning(f"Cannot exit {symbol} - not in active trades")
            return False

        trade_info = self.active_trades[symbol]

        try:
            # Cancel existing GTT orders
            sl_gtt_id = trade_info.get("sl_gtt_id")
            tp_gtt_id = trade_info.get("tp_gtt_id")

            if sl_gtt_id:
                if sl_gtt_id.startswith("order:"):
                    # Cancel regular order
                    order_id = sl_gtt_id.split(":")[1]
                    self.kite_client.cancel_order(variety="regular", order_id=order_id)
                    logger.info(f"Cancelled stop-loss order {order_id} for {symbol}")
                else:
                    # Delete GTT
                    self.kite_client.delete_gtt(trigger_id=int(sl_gtt_id))
                    logger.info(f"Cancelled stop-loss GTT {sl_gtt_id} for {symbol}")

            if tp_gtt_id:
                if tp_gtt_id.startswith("order:"):
                    # Cancel regular order
                    order_id = tp_gtt_id.split(":")[1]
                    self.kite_client.cancel_order(variety="regular", order_id=order_id)
                    logger.info(f"Cancelled target order {order_id} for {symbol}")
                else:
                    # Delete GTT
                    self.kite_client.delete_gtt(trigger_id=int(tp_gtt_id))
                    logger.info(f"Cancelled target GTT {tp_gtt_id} for {symbol}")

            # Place market sell order
            quantity = trade_info.get("quantity", 0)
            product_type = trade_info.get("product_type", ORDER_TYPE)

            order_params = {
                "exchange": OPTION_EXCHANGE,
                "tradingsymbol": symbol,
                "transaction_type": "SELL",
                "quantity": quantity,
                "product": product_type,
                "order_type": "MARKET",
            }

            # Place the exit order
            order_id = self.kite_client.place_order(variety="regular", **order_params)
            logger.trade(f"EXIT order placed for {symbol}. Order ID: {order_id}. Reason: {reason}")

            # Update trade info
            trade_info["exit_type"] = reason
            trade_info["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade_info["exit_order_id"] = order_id

            # Save trade history before removing from active trades
            # Here you might want to add code to save completed trades to a database or file

            # Remove from active trades
            self.active_trades.pop(symbol, None)

            return True

        except Exception as e:
            logger.error(f"Error exiting position for {symbol}: {e}")
            return False

    def exit_all_positions(self, reason="Manual exit"):
        """
        Exit all active positions.

        Args:
            reason: Reason for exiting positions

        Returns:
            int: Number of positions exited
        """
        if not self.active_trades:
            logger.info("No active trades to exit")
            return 0

        count = 0
        for symbol, trade_info in list(self.active_trades.items()):
            if trade_info.get("status") == "ACTIVE":
                if self.exit_position(symbol, reason):
                    count += 1

        logger.info(f"Exited {count} positions. Reason: {reason}")
        return count

    def log_positions(self):
        """
        Log current positions with details.
        """
        try:
            positions = self.kite_client.positions()

            # Track if we found any positions
            found_positions = False

            # Check day positions
            if "day" in positions and positions["day"]:
                day_positions = [pos for pos in positions["day"] if pos.get("quantity", 0) != 0]
                if day_positions:
                    found_positions = True
                    logger.info("\n" + "-" * 50)
                    logger.info(f"POSITIONS: {len(day_positions)} day positions")
                    logger.info("-" * 50)

                    for pos in day_positions:
                        symbol = pos.get("tradingsymbol", "Unknown")
                        entry_price = pos.get("average_price", 0)
                        quantity = pos.get("quantity", 0)
                        pnl = pos.get("pnl", 0)

                        # Format based on profit/loss
                        if pnl > 0:
                            pnl_str = f"+INR {pnl:.2f}"
                        else:
                            pnl_str = f"-INR {abs(pnl):.2f}"

                        logger.info(f"Position - {symbol}: Qty {quantity}, Entry {entry_price:.2f}, P/L {pnl_str}")

            # Also check net positions
            if "net" in positions and positions["net"]:
                net_positions = [pos for pos in positions["net"] if pos.get("quantity", 0) != 0]
                if net_positions and not found_positions:
                    found_positions = True
                    logger.info("\n" + "-" * 50)
                    logger.info(f"POSITIONS: {len(net_positions)} net positions")
                    logger.info("-" * 50)

                    for pos in net_positions:
                        symbol = pos.get("tradingsymbol", "Unknown")
                        entry_price = pos.get("average_price", 0)
                        quantity = pos.get("quantity", 0)
                        pnl = pos.get("pnl", 0)

                        # Format based on profit/loss
                        if pnl > 0:
                            pnl_str = f"+INR {pnl:.2f}"
                        else:
                            pnl_str = f"-INR {abs(pnl):.2f}"

                        logger.info(f"Position - {symbol}: Qty {quantity}, Entry {entry_price:.2f}, P/L {pnl_str}")

            if not found_positions:
                logger.info("No open positions")

            # Log pending orders if any
            try:
                orders = self.kite_client.orders()
                if orders:
                    pending_orders = [o for o in orders if o.get("status") in ["OPEN", "PENDING", "TRIGGER PENDING"]]
                    if pending_orders:
                        logger.info("\n" + "-" * 50)
                        logger.info(f"PENDING ORDERS: {len(pending_orders)}")
                        logger.info("-" * 50)
                        for order in pending_orders:
                            order_status = order.get("status", "")
                            logger.info(f"Order {order.get('order_id')}: {order.get('tradingsymbol')} - "
                                        f"{order.get('transaction_type')} {order.get('quantity')} @ {order.get('price')} - "
                                        f"{order_status}")
            except Exception as e:
                logger.error(f"Error fetching orders: {e}")

        except Exception as e:
            logger.error(f"Error logging positions: {e}")