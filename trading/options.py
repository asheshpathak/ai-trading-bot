import logging
from datetime import datetime, timedelta
from models.technical import get_atm_strike
from config.settings import OPTION_EXCHANGE

# Initialize logger
logger = logging.getLogger("TradingBot")


class OptionsFinder:
    def __init__(self, kite_client):
        """
        Initialize with KiteClient instance.

        Args:
            kite_client: KiteClient instance
        """
        self.kite_client = kite_client

    def get_nearest_option_expiry(self, underlying):
        """
        Find the nearest option expiry for a symbol.

        Args:
            underlying: Underlying symbol (e.g. 'NIFTY', 'BANKNIFTY', 'SBIN')

        Returns:
            datetime.date: Expiry date or None if not found
        """
        try:
            instruments = self.kite_client.get_cached_instruments(OPTION_EXCHANGE)
            if not instruments:
                logger.error(f"No instruments returned for {OPTION_EXCHANGE}")
                return None

            logger.info(f"Got {len(instruments)} instruments from {OPTION_EXCHANGE}")

            # Manual symbol cleaning
            underlying_clean = underlying.strip().upper()

            # Remove exchange prefix if present
            if ':' in underlying_clean:
                underlying_clean = underlying_clean.split(':', 1)[1]

            # Remove -EQ suffix if present
            if underlying_clean.endswith('-EQ'):
                underlying_clean = underlying_clean[:-3]

            logger.info(f"Looking for option expiries for {underlying_clean}")

            # Filter instruments for this underlying
            filtered = []
            for inst in instruments:
                inst_name = inst.get("name", "").strip().upper()
                inst_symbol = inst.get("tradingsymbol", "").strip().upper()

                # Match on name or as part of trading symbol
                if ((inst_name == underlying_clean or
                     underlying_clean in inst_symbol) and
                        inst.get("instrument_type") in ["CE", "PE"]):
                    filtered.append(inst)

            if not filtered:
                # Try more flexible matching
                logger.warning(f"No exact matches found for {underlying_clean}. Trying flexible matching...")
                for inst in instruments:
                    inst_name = inst.get("name", "").strip().upper()
                    inst_symbol = inst.get("tradingsymbol", "").strip().upper()

                    # Try partial matches
                    if (inst.get("instrument_type") in ["CE", "PE"] and
                            (inst_symbol.startswith(underlying_clean) or
                             underlying_clean.startswith(inst_name) or
                             inst_name.startswith(underlying_clean))):
                        logger.info(f"Found flexible match: {inst_symbol} (name: {inst_name})")
                        filtered.append(inst)

            if not filtered:
                logger.warning(f"No options found for {underlying_clean}")
                return None

            logger.info(f"Found {len(filtered)} options for {underlying_clean}")

            # Get all valid expiry dates
            today = datetime.now().date()
            logger.info(f"Today's date: {today}")
            expiries = set()

            # Process all instruments to find valid expiries
            for inst in filtered:
                try:
                    # Handle both string and datetime.date formats for expiry
                    expiry_date = inst["expiry"]

                    # Convert to date object if it's a string
                    if isinstance(expiry_date, str):
                        expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()

                    # Check if the expiry is today or in the future
                    if expiry_date >= today:
                        expiries.add(expiry_date)
                except Exception as e:
                    logger.debug(f"Error processing expiry: {e}")
                    continue

            if not expiries:
                logger.warning(f"No valid expiries found for {underlying_clean}")
                return None

            # Sort expiries and get the nearest one
            sorted_expiries = sorted(expiries)
            nearest_expiry = sorted_expiries[0]

            # If today is expiry day, use next expiry if available
            if nearest_expiry == today and len(sorted_expiries) > 1:
                nearest_expiry = sorted_expiries[1]
                logger.info(f"Today is expiry day. Using next expiry: {nearest_expiry}")

            logger.info(f"Selected expiry for {underlying_clean}: {nearest_expiry}")
            return nearest_expiry

        except Exception as e:
            logger.error(f"Error finding option expiry: {e}")
            return None

    def get_option_for_strike(self, underlying, expiry, strike, option_type):
        """
        Find an option contract matching the criteria.

        Args:
            underlying: Underlying symbol
            expiry: Expiry date
            strike: Strike price
            option_type: Option type ('CE' or 'PE')

        Returns:
            dict: Option instrument or None if not found
        """
        try:
            instruments = self.kite_client.get_cached_instruments(OPTION_EXCHANGE)
            if not instruments:
                logger.error(f"No instruments available for {OPTION_EXCHANGE}")
                return None

            # Manual symbol cleaning
            underlying_clean = underlying.strip().upper()

            # Remove exchange prefix if present
            if ':' in underlying_clean:
                underlying_clean = underlying_clean.split(':', 1)[1]

            # Remove -EQ suffix if present
            if underlying_clean.endswith('-EQ'):
                underlying_clean = underlying_clean[:-3]

            # Ensure expiry is a datetime.date object
            expiry_date = expiry
            if isinstance(expiry, str):
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()

            # Format the target expiry as a string for logging
            expiry_str = str(expiry_date)
            logger.info(
                f"Looking for {underlying_clean} {option_type} options with strike {strike} expiring on {expiry_str}")

            # Find matches using both name and tradingsymbol
            matching_instruments = []

            for inst in instruments:
                inst_name = inst.get("name", "").strip().upper()
                inst_symbol = inst.get("tradingsymbol", "").strip().upper()
                inst_type = inst.get("instrument_type")

                # Match based on name field and option type
                name_match = inst_name == underlying_clean and inst_type == option_type

                # Match based on tradingsymbol containing the underlying and option type
                symbol_match = (underlying_clean in inst_symbol and
                                (inst_type == option_type or
                                 (inst_type == "" and
                                  ((option_type == "CE" and "CE" in inst_symbol) or
                                   (option_type == "PE" and "PE" in inst_symbol)))))

                if name_match or symbol_match:
                    try:
                        # Handle both string and datetime.date formats for expiry
                        inst_expiry = inst["expiry"]
                        if isinstance(inst_expiry, str):
                            inst_expiry = datetime.strptime(inst_expiry, "%Y-%m-%d").date()

                        # Compare as date objects
                        if inst_expiry == expiry_date and abs(float(inst["strike"]) - float(strike)) < 1e-3:
                            matching_instruments.append(inst)
                    except Exception as e:
                        logger.debug(f"Error processing instrument: {e}")
                        continue

            # If we have matches, return the first one
            if matching_instruments:
                logger.info(
                    f"Found {len(matching_instruments)} matching instruments for {underlying_clean} {option_type}")
                best_match = matching_instruments[0]
                logger.info(f"Selected option: {best_match.get('tradingsymbol')}")
                return best_match

            # If no matches found with exact strike, try to find the closest available strike
            all_strikes = []
            for inst in instruments:
                inst_name = inst.get("name", "").strip().upper()
                inst_symbol = inst.get("tradingsymbol", "").strip().upper()
                inst_type = inst.get("instrument_type")

                # Using both name and symbol matching for flexibility
                name_match = inst_name == underlying_clean and inst_type == option_type
                symbol_match = (underlying_clean in inst_symbol and
                                (inst_type == option_type or
                                 (inst_type == "" and
                                  ((option_type == "CE" and "CE" in inst_symbol) or
                                   (option_type == "PE" and "PE" in inst_symbol)))))

                if name_match or symbol_match:
                    try:
                        # Handle both string and datetime.date formats for expiry
                        inst_expiry = inst["expiry"]
                        if isinstance(inst_expiry, str):
                            inst_expiry = datetime.strptime(inst_expiry, "%Y-%m-%d").date()

                        if inst_expiry == expiry_date:
                            all_strikes.append((float(inst["strike"]), inst))
                    except Exception as e:
                        logger.debug(f"Error processing strike: {e}")
                        continue

            if all_strikes:
                # Find the closest strike
                all_strikes.sort(key=lambda x: abs(x[0] - float(strike)))
                closest_strike, closest_inst = all_strikes[0]
                logger.info(
                    f"No exact strike match. Using closest available strike: {closest_strike} (requested: {strike})")
                return closest_inst

            logger.warning(
                f"No matching instruments found for {underlying_clean} {option_type} options with strike near {strike}")
            return None

        except Exception as e:
            logger.error(f"Error finding option for strike: {e}")
            return None

    def get_option_tradingsymbol(self, underlying, recommendation, live_price):
        """
        Find appropriate option based on signal.

        Args:
            underlying: Underlying symbol
            recommendation: Signal ('Buy Call Option' or 'Buy Put Option')
            live_price: Current price of underlying

        Returns:
            tuple: (option_tradingsymbol, lot_size) or (None, None) if not found
        """
        try:
            atm_strike = get_atm_strike(live_price)
            logger.info(f"Looking for {underlying} options with ATM strike {atm_strike}")

            # Get the nearest expiry
            expiry = self.get_nearest_option_expiry(underlying)
            if expiry is None:
                logger.warning(f"Could not find any valid expiry for {underlying}")
                return None, None

            # Determine option type based on recommendation
            option_type = "CE" if recommendation == "Buy Call Option" else "PE"

            # Try to find the exact ATM strike
            logger.info(f"Looking for {underlying} {option_type} options with strike {atm_strike}")
            opt_inst = self.get_option_for_strike(underlying, expiry, atm_strike, option_type)

            # If exact ATM strike not found, try stepping up or down
            if opt_inst is None:
                logger.info(f"Could not find exact ATM strike {atm_strike}, trying nearby strikes")
                # Try +/- 1 step
                for step in [50, 100, 150, 200]:
                    # Try higher strike
                    higher_strike = atm_strike + step
                    logger.debug(f"Trying higher strike {higher_strike}")
                    opt_inst = self.get_option_for_strike(underlying, expiry, higher_strike, option_type)
                    if opt_inst:
                        logger.info(f"Found option with higher strike {higher_strike}")
                        break

                    # Try lower strike
                    lower_strike = atm_strike - step
                    logger.debug(f"Trying lower strike {lower_strike}")
                    opt_inst = self.get_option_for_strike(underlying, expiry, lower_strike, option_type)
                    if opt_inst:
                        logger.info(f"Found option with lower strike {lower_strike}")
                        break

            # If we found a matching option, return its details
            if opt_inst:
                lot_size = opt_inst.get("lot_size", 1)
                logger.info(
                    f"Found {underlying} {option_type} option {opt_inst['tradingsymbol']} with lot size {lot_size}")
                return opt_inst["tradingsymbol"], lot_size
            else:
                logger.warning(f"Could not find any suitable {option_type} options for {underlying}")
                return None, None

        except Exception as e:
            logger.error(f"Error getting option tradingsymbol: {e}")
            return None, None