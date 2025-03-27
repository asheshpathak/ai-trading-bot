import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# Initialize logger
logger = logging.getLogger("TradingBot")


def fetch_historical_data(client, token, from_date, to_date, interval="minute"):
    """
    Fetches historical data from KiteConnect.

    Args:
        client: KiteClient instance
        token: Instrument token
        from_date: Start date (datetime or str)
        to_date: End date (datetime or str)
        interval: Data interval (minute, day, etc.)

    Returns:
        pandas.DataFrame with historical data or None if error
    """
    try:
        # Format dates properly if they're datetime objects
        if isinstance(from_date, datetime):
            from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            from_date_str = from_date

        if isinstance(to_date, datetime):
            to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            to_date_str = to_date

        logger.debug(f"Fetching {interval} data from {from_date_str} to {to_date_str}")

        # Fetch data
        data = client.historical_data(token, from_date_str, to_date_str, interval)

        if not data:
            logger.warning("No data returned from API")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            logger.debug(f"Successfully fetched {len(df)} data points")
        return df

    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None


def fetch_minute_data(client, token, minutes=30):
    """
    Fetches minute-level historical data for the specified number of minutes.

    Args:
        client: KiteClient instance
        token: Instrument token
        minutes: Number of minutes of data to fetch

    Returns:
        pandas.DataFrame with minute data or None if error
    """
    try:
        now = datetime.now()
        from_time = now - timedelta(minutes=minutes)

        # Format dates properly
        from_time_str = from_time.strftime("%Y-%m-%d %H:%M:%S")
        to_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"Fetching minute data from {from_time_str} to {to_time_str}")

        return fetch_historical_data(client, token, from_time_str, to_time_str, "minute")

    except Exception as e:
        logger.error(f"Error fetching minute data: {e}")
        return None


def fetch_previous_close(client, token):
    """
    Fetches previous day's closing price for a symbol.

    Args:
        client: KiteClient instance
        token: Instrument token

    Returns:
        float: Previous close price or None if error
    """
    try:
        today = datetime.now()
        from_date = today - timedelta(days=5)  # Increased to 5 days to better handle weekends/holidays
        data = client.historical_data(token, from_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), "day")

        if len(data) >= 2:
            return data[-2]["close"]
        elif len(data) == 1:
            return data[0]["close"]
        else:
            return None

    except Exception as e:
        logger.error(f"Error fetching previous close: {e}")
        return None


def fetch_live_quote(client, symbol, exchange="NSE"):
    """
    Fetches the live quote for a symbol.

    Args:
        client: KiteClient instance
        symbol: Symbol to fetch quote for
        exchange: Exchange code

    Returns:
        float: Last price or None if error
    """
    try:
        # Manual symbol cleaning
        clean_symbol_str = symbol.strip().upper()

        # Remove exchange prefix if present
        if ':' in clean_symbol_str:
            clean_symbol_str = clean_symbol_str.split(':', 1)[1]

        instrument_key = f"{exchange}:{clean_symbol_str}"

        # Get quote
        quote = client.quote(instrument_key)
        return quote[instrument_key]["last_price"]

    except Exception as e:
        logger.error(f"Error fetching live quote for {symbol}: {e}")
        return None


def update_historical_data(client, token, symbol, output_file, days=30):
    """
    Updates the CSV file with historical minute data.

    Args:
        client: KiteClient instance
        token: Instrument token
        symbol: Symbol for logging
        output_file: Path to output CSV file
        days: Number of days of historical data to fetch if file doesn't exist

    Returns:
        pandas.DataFrame with updated data or None if error
    """
    now = datetime.now()

    if os.path.exists(output_file):
        try:
            logger.info(f"Reading existing data from {output_file}")
            df_existing = pd.read_csv(output_file, parse_dates=["date"])
            df_existing.set_index("date", inplace=True)

            if df_existing.empty:
                logger.info(f"Existing file {output_file} is empty. Fetching {days} days of data.")
                from_date = now - timedelta(days=days)
                to_date = now
                df_updated = fetch_historical_data(client, token, from_date, to_date, "minute")
            else:
                last_timestamp = df_existing.index.max()
                logger.info(f"Last timestamp in file: {last_timestamp}")

                # Make sure we have at least 1 minute gap
                from_date = last_timestamp + timedelta(minutes=1)

                # If last_timestamp is very old (more than 60 days), limit to last 30 days
                days_difference = (now - last_timestamp).days
                if days_difference > 60:
                    logger.info(f"Last timestamp is {days_difference} days old. Limiting to last 30 days.")
                    from_date = now - timedelta(days=30)

                to_date = now
                logger.info(f"Fetching new data for {symbol} from {from_date} to {to_date}")

                df_new = fetch_historical_data(client, token, from_date, to_date, "minute")
                if df_new is not None and not df_new.empty:
                    logger.info(
                        f"Concatenating existing data ({len(df_existing)} rows) with new data ({len(df_new)} rows)")
                    df_updated = pd.concat([df_existing, df_new]).sort_index()
                    # Remove duplicate indices if any
                    df_updated = df_updated[~df_updated.index.duplicated(keep='last')]
                else:
                    logger.info("No new data to add. Using existing data.")
                    df_updated = df_existing
        except Exception as e:
            logger.error(f"Error reading/updating {output_file}: {e}")
            logger.info(f"Starting fresh with last {days} days of data.")
            from_date = now - timedelta(days=days)
            to_date = now
            df_updated = fetch_historical_data(client, token, from_date, to_date, "minute")
    else:
        logger.info(f"File {output_file} does not exist. Creating new file with last {days} days of data.")
        from_date = now - timedelta(days=days)
        to_date = now
        df_updated = fetch_historical_data(client, token, from_date, to_date, "minute")

    if df_updated is not None and not df_updated.empty:
        logger.info(f"Saving {len(df_updated)} rows to {output_file}")
        df_updated.to_csv(output_file)
        return df_updated
    else:
        logger.warning(f"No data to save to {output_file}")
        return None