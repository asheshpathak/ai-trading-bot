import os
import logging
import pickle
from config.settings import PICKLES_DIR
from models.technical import prepare_features_for_prediction

# Initialize logger
logger = logging.getLogger("TradingBot")


class MLPredictor:
    def __init__(self, pickles_dir=PICKLES_DIR):
        """
        Initialize predictor with path to model pickle files.

        Args:
            pickles_dir: Directory containing trained model pickle files
        """
        self.pickles_dir = pickles_dir
        self.loaded_models = {}

    def load_model(self, symbol):
        """
        Load a trained model for a specific symbol.

        Args:
            symbol: Symbol to load model for

        Returns:
            tuple: (model, scaler, feature_cols) or None if model not found
        """
        try:
            # Clean symbol for file lookup
            symbol_clean = symbol.strip().upper()
            if symbol_clean.endswith("-EQ"):
                symbol_clean = symbol_clean[:-3]

            model_file = os.path.join(self.pickles_dir, f"{symbol_clean}_scalping_model.pkl")

            # Check if model is already loaded
            if symbol_clean in self.loaded_models:
                logger.debug(f"Using cached model for {symbol_clean}")
                return self.loaded_models[symbol_clean]

            # Load from file if it exists
            if os.path.exists(model_file):
                try:
                    with open(model_file, "rb") as f:
                        model_data = pickle.load(f)
                        self.loaded_models[symbol_clean] = model_data
                        logger.info(f"Loaded ML model for {symbol_clean}")
                        return model_data
                except Exception as e:
                    logger.error(f"Error loading model for {symbol_clean}: {e}")
                    return None
            else:
                logger.debug(f"Model file {model_file} not found for {symbol_clean}")
                return None
        except Exception as e:
            logger.error(f"Error in load_model: {e}")
            return None

    def predict(self, symbol, minute_df):
        """
        Make a prediction using the trained model for a symbol.

        Args:
            symbol: Symbol to predict for
            minute_df: DataFrame with minute-level OHLCV data

        Returns:
            tuple: (signal, probability)
        """
        try:
            # Load model
            model_data = self.load_model(symbol)
            if not model_data:
                logger.warning(f"No model available for {symbol}")
                return None, 0

            model, scaler, feature_cols = model_data

            # Prepare features
            prepared_df = prepare_features_for_prediction(minute_df)
            if prepared_df is None or prepared_df.empty:
                logger.error(f"No data available after preparing features for {symbol}")
                return None, 0

            # Verify all required features exist
            missing_cols = [col for col in feature_cols if col not in prepared_df.columns]
            if missing_cols:
                logger.error(f"Missing columns in data for {symbol}: {missing_cols}")
                return None, 0

            # Get the latest row for prediction
            try:
                latest_features = prepared_df[feature_cols].iloc[-1].values.reshape(1, -1)
                X_scaled = scaler.transform(latest_features)
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][pred]

                # Convert numeric prediction to signal
                signal = "Buy Call Option" if pred == 1 else "Buy Put Option"
                logger.debug(f"ML prediction for {symbol}: {signal} with confidence {prob:.2f}")

                return signal, prob
            except Exception as e:
                logger.error(f"Error making ML prediction for {symbol}: {e}")
                return None, 0
        except Exception as e:
            logger.error(f"Error in prediction pipeline for {symbol}: {e}")
            return None, 0

    def load_all_models(self):
        """
        Load all available models from the pickles directory.

        Returns:
            dict: Dictionary of loaded models by symbol
        """
        try:
            # Get all pickle files
            if not os.path.exists(self.pickles_dir):
                logger.warning(f"Pickles directory {self.pickles_dir} not found")
                return {}

            model_files = [f for f in os.listdir(self.pickles_dir) if f.endswith("_scalping_model.pkl")]
            logger.info(f"Found {len(model_files)} model files in {self.pickles_dir}")

            # Load each model
            for model_file in model_files:
                try:
                    # Extract symbol from filename
                    symbol = model_file.replace("_scalping_model.pkl", "")

                    # Load the model
                    if symbol not in self.loaded_models:
                        model_path = os.path.join(self.pickles_dir, model_file)
                        with open(model_path, "rb") as f:
                            self.loaded_models[symbol] = pickle.load(f)
                        logger.debug(f"Loaded model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading model from {model_file}: {e}")
                    continue

            logger.info(f"Successfully loaded {len(self.loaded_models)} models")
            return self.loaded_models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {}