import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.technical import prepare_features_for_training

# Initialize logger
logger = logging.getLogger("TradingBot")


class ModelTrainer:
    def __init__(self, training_data_dir, pickles_dir):
        """
        Initialize trainer with paths to data and output directories.

        Args:
            training_data_dir: Directory with training data CSV files
            pickles_dir: Directory to save trained model pickle files
        """
        self.training_data_dir = training_data_dir
        self.pickles_dir = pickles_dir

        # Ensure directories exist
        os.makedirs(training_data_dir, exist_ok=True)
        os.makedirs(pickles_dir, exist_ok=True)

    def train_model(self, df, symbol):
        """
        Train a RandomForest model for a symbol.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol for logging and model saving

        Returns:
            tuple: (model, scaler, feature_cols) or None if error
        """
        try:
            logger.info(f"Preparing features for {symbol}...")
            df_prepared = prepare_features_for_training(df)

            if df_prepared is None or df_prepared.empty:
                logger.warning(f"Skipping {symbol}: Not enough data after feature preparation")
                return None

            # Features to use for prediction
            feature_cols = ["returns", "sma20", "ema20", "macd", "macd_signal", "rsi"]

            # Check if all features exist
            missing_features = [col for col in feature_cols if col not in df_prepared.columns]
            if missing_features:
                logger.error(f"Missing features for {symbol}: {missing_features}")
                return None

            X = df_prepared[feature_cols].values
            y = df_prepared["target"].values

            # Optional: Split into train/test for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            logger.info(f"Model for {symbol} - Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")

            # Retrain on all data for final model
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)

            return model, scaler, feature_cols

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return None

    def save_model(self, model_data, symbol):
        """
        Save a trained model to pickle file.

        Args:
            model_data: Tuple of (model, scaler, feature_cols)
            symbol: Symbol for filename

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_data is None:
                logger.warning(f"No model data to save for {symbol}")
                return False

            # Clean symbol for file naming
            symbol_clean = symbol.strip().upper()
            if symbol_clean.endswith("-EQ"):
                symbol_clean = symbol_clean[:-3]

            output_file = os.path.join(self.pickles_dir, f"{symbol_clean}_scalping_model.pkl")

            with open(output_file, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model for {symbol_clean} saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
            return False

    def load_training_data(self, symbol):
        """
        Load training data for a symbol from CSV file.

        Args:
            symbol: Symbol to load data for

        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Clean symbol for file lookup
            symbol_clean = symbol.strip().upper()
            if symbol_clean.endswith("-EQ"):
                symbol_clean = symbol_clean[:-3]

            csv_filename = os.path.join(self.training_data_dir, f"{symbol_clean}_minute_data.csv")

            if not os.path.exists(csv_filename):
                logger.warning(f"No training data file found for {symbol_clean}: {csv_filename}")
                return None

            # Load and parse data
            df = pd.read_csv(csv_filename, parse_dates=["date"])
            df.set_index("date", inplace=True)

            if df.empty:
                logger.warning(f"Empty training data file for {symbol_clean}")
                return None

            logger.info(f"Loaded {len(df)} data points for {symbol_clean}")
            return df

        except Exception as e:
            logger.error(f"Error loading training data for {symbol}: {e}")
            return None

    def train_and_save_model(self, symbol):
        """
        Complete flow to load data, train model, and save for a symbol.

        Args:
            symbol: Symbol to process

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load data
            df = self.load_training_data(symbol)
            if df is None:
                return False

            # Train model
            model_data = self.train_model(df, symbol)
            if model_data is None:
                return False

            # Save model
            return self.save_model(model_data, symbol)

        except Exception as e:
            logger.error(f"Error in train_and_save_model for {symbol}: {e}")
            return False