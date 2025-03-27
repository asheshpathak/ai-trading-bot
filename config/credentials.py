import os
from dotenv import load_dotenv
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class Credentials:
    def __init__(self):
        # Try to load from environment variables first
        load_dotenv()

        self.api_key = os.getenv('ZERODHA_API_KEY')
        self.api_secret = os.getenv('ZERODHA_API_SECRET')

        # If env vars aren't set, try to load from encrypted file
        if not self.api_key or not self.api_secret:
            self._load_from_encrypted_file()

    def _load_from_encrypted_file(self):
        """Load credentials from an encrypted file using a master password"""
        try:
            # The master password could be stored in an env var or prompted from user
            master_password = os.getenv('ZERODHA_MASTER_PASSWORD')

            if not master_password:
                # If not in env var, prompt the user (only when running interactively)
                import getpass
                master_password = getpass.getpass("Enter master password: ")

            if not os.path.exists('config/credentials.enc'):
                # If the encrypted file doesn't exist yet, create it
                self._create_encrypted_file(master_password)
            else:
                # Otherwise decrypt existing file
                creds = self._decrypt_credentials(master_password)
                self.api_key = creds.get('api_key')
                self.api_secret = creds.get('api_secret')

        except Exception as e:
            print(f"Error loading credentials: {e}")
            raise

    def _get_key_from_password(self, password):
        """Derive encryption key from password"""
        # Use a static salt (could be improved by storing salt separately)
        salt = b'zerodha_trading_bot_salt'

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _encrypt_credentials(self, master_password, creds_dict):
        """Encrypt credentials using the master password"""
        key = self._get_key_from_password(master_password)
        f = Fernet(key)

        # Convert dict to string and encrypt
        creds_str = f"{creds_dict['api_key']}:{creds_dict['api_secret']}"
        encrypted_data = f.encrypt(creds_str.encode())

        return encrypted_data

    def _decrypt_credentials(self, master_password):
        """Decrypt credentials using the master password"""
        key = self._get_key_from_password(master_password)
        f = Fernet(key)

        # Read encrypted data from file
        with open('config/credentials.enc', 'rb') as file:
            encrypted_data = file.read()

        # Decrypt data
        decrypted_data = f.decrypt(encrypted_data).decode()
        api_key, api_secret = decrypted_data.split(':')

        return {'api_key': api_key, 'api_secret': api_secret}

    def _create_encrypted_file(self, master_password):
        """Create encrypted credentials file (run once when setting up)"""
        # Prompt for API credentials
        import getpass
        print("No encrypted credentials file found. Creating one...")
        api_key = input("Enter Zerodha API Key: ")
        api_secret = getpass.getpass("Enter Zerodha API Secret: ")

        # Encrypt and save
        creds = {'api_key': api_key, 'api_secret': api_secret}
        encrypted_data = self._encrypt_credentials(master_password, creds)

        # Ensure directory exists
        os.makedirs('config', exist_ok=True)

        # Write to file
        with open('config/credentials.enc', 'wb') as file:
            file.write(encrypted_data)

        # Set the values
        self.api_key = api_key
        self.api_secret = api_secret

        print("Credentials encrypted and saved successfully.")

    def get_credentials(self):
        """Return credentials as a dictionary"""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret
        }