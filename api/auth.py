import os
import time
import threading
import ssl
import logging
from datetime import datetime, timedelta
import pytz
from flask import Flask, request
from kiteconnect import KiteConnect

# Initialize logger
logger = logging.getLogger("TradingBot")

# Callback configuration
CALLBACK_HOST = "localhost"
CALLBACK_PORT = 5000
CALLBACK_PATH = "/redirect"
CALLBACK_URL = f"https://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

# Files for authentication
ACCESS_TOKEN_FILE = "access_token.txt"
TOKEN_TIMESTAMP_FILE = "token_timestamp.txt"
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

# Global variable to track if server is running
flask_server_running = False

# Flask app for OAuth callback
app = Flask(__name__)
token_holder = {"token": None}


@app.route(CALLBACK_PATH)
def oauth_callback():
    """
    Zerodha will redirect to this endpoint after a successful login.
    Captures the request token, generates the session, and saves the access token.
    """
    req_token = request.args.get("request_token")
    if req_token:
        try:
            logger.info(f"Got request token: {req_token}")
            kite = KiteConnect(api_key=token_holder["api_key"])
            session_data = kite.generate_session(req_token, api_secret=token_holder["api_secret"])
            access_token = session_data["access_token"]
            token_holder["token"] = access_token

            # Save token to file
            with open(ACCESS_TOKEN_FILE, "w") as f:
                f.write(access_token)

            # Save timestamp
            with open(TOKEN_TIMESTAMP_FILE, "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            logger.info("Access token generated and saved successfully.")
            return """
            <html>
                <body style='font-family: Arial, sans-serif; text-align: center; padding: 50px;'>
                    <h2 style='color: #4CAF50;'>Access token generated successfully!</h2>
                    <p>You can close this window and return to the trading bot.</p>
                </body>
            </html>
            """
        except Exception as e:
            logger.error(f"Error generating session: {e}")
            return f"""
            <html>
                <body style='font-family: Arial, sans-serif; text-align: center; padding: 50px;'>
                    <h2 style='color: #F44336;'>Error generating session</h2>
                    <p>{str(e)}</p>
                    <p>Please try again.</p>
                </body>
            </html>
            """
    else:
        return """
        <html>
            <body style='font-family: Arial, sans-serif; text-align: center; padding: 50px;'>
                <h2 style='color: #F44336;'>No request token received</h2>
                <p>No request_token parameter was found in the URL.</p>
                <p>Please try the login process again.</p>
            </body>
        </html>
        """


def generate_ssl_cert():
    """Generate self-signed SSL certificates for HTTPS if they don't exist."""
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        logger.info("Generating self-signed SSL certificates for HTTPS...")
        try:
            from OpenSSL import crypto

            # Create key pair
            k = crypto.PKey()
            k.generate_key(crypto.TYPE_RSA, 2048)

            # Create self-signed certificate
            cert = crypto.X509()
            cert.get_subject().CN = CALLBACK_HOST
            cert.set_serial_number(1000)
            cert.gmtime_adj_notBefore(0)
            cert.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)  # 10 years
            cert.set_issuer(cert.get_subject())
            cert.set_pubkey(k)
            cert.sign(k, 'sha256')

            # Save certificate and key files
            with open(CERT_FILE, "wb") as f:
                f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            with open(KEY_FILE, "wb") as f:
                f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

            logger.info(f"SSL certificates generated: {CERT_FILE} and {KEY_FILE}")
        except Exception as e:
            logger.error(f"Failed to generate SSL certificates: {e}")
            logger.error("Please install pyOpenSSL: pip install pyOpenSSL")
            logger.error("Or manually create SSL certificates using OpenSSL.")
            raise


def run_flask_server():
    """Run an HTTPS server with self-signed certificates for OAuth callback."""
    global flask_server_running

    if flask_server_running:
        return

    # Generate certificates if needed
    generate_ssl_cert()

    try:
        # Create SSL context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(CERT_FILE, KEY_FILE)

        # Mark server as running
        flask_server_running = True

        # Run Flask with SSL
        app.run(host="0.0.0.0", port=CALLBACK_PORT, ssl_context=context, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")
        flask_server_running = False


def is_token_valid():
    """
    Checks if the access token exists and is not expired.
    Zerodha tokens typically expire at 6 AM IST the next day.
    """
    if not os.path.exists(ACCESS_TOKEN_FILE) or not os.path.exists(TOKEN_TIMESTAMP_FILE):
        return False

    try:
        with open(TOKEN_TIMESTAMP_FILE, "r") as f:
            timestamp_str = f.read().strip()
            token_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

        # Check if token was generated today and it's before 6 AM IST tomorrow
        india_tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(india_tz)
        token_time = token_time.replace(tzinfo=india_tz)
        expiry_time = (token_time.replace(hour=6, minute=0, second=0) + timedelta(days=1))

        return now < expiry_time
    except Exception as e:
        logger.error(f"Error checking token validity: {e}")
        return False


def test_token(api_key, token):
    """Tests if the token is valid by making a simple API call."""
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        profile = kite.profile()  # This will fail if token is invalid
        return True
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        return False


def get_access_token(api_key, api_secret):
    """
    Gets a valid access token, either from file or by initiating the login process.
    """
    global token_holder

    # Store API credentials in token_holder for OAuth callback
    token_holder["api_key"] = api_key
    token_holder["api_secret"] = api_secret

    # Check if we have a valid token already
    if is_token_valid():
        with open(ACCESS_TOKEN_FILE, "r") as f:
            token = f.read().strip()

        # Verify token works
        if test_token(api_key, token):
            logger.info("Using existing valid access token.")
            return token
        else:
            logger.warning("Existing token failed validation.")

    # Start the Flask server in a separate thread if not already running
    if not flask_server_running:
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        time.sleep(1)  # Give the server a moment to start

    # Reset token holder
    token_holder["token"] = None

    # Generate login URL
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    # Display login instructions
    logger.info("\n" + "=" * 80)
    logger.info("ACCESS TOKEN REQUIRED")
    logger.info("=" * 80)
    logger.info("Please open the following URL in your browser and log in:")
    logger.info(login_url)
    logger.info(f"After logging in, you will be redirected to {CALLBACK_URL}")
    logger.info("Note: You may need to accept the self-signed certificate in your browser.")
    logger.info("=" * 80 + "\n")

    # Wait for token with timeout
    timeout = 300  # 5 minutes
    start_time = time.time()
    while token_holder["token"] is None:
        if time.time() - start_time > timeout:
            logger.error("Timeout waiting for access token.")
            raise Exception("Timeout waiting for access token")
        time.sleep(1)

    return token_holder["token"]