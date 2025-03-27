#!/usr/bin/env python3
"""
Enhanced Model Training Script for Trading Bot

This script trains machine learning models for each symbol using enhanced features:
- Additional technical indicators (ADX, VWAP bands)
- Momentum features
- Volume analysis
- Support for parallelized training

The trained models are used by the trading bot to generate trading signals.
"""

import os
import argparse
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

from config.settings import TRAINING_DATA_DIR, PICKLES_DIR, SYMBOLS_DIR, DAYS_OF_HISTORICAL_DATA
from config.credentials import Credentials
from api.client import KiteClient
from api.data import update_historical_data
from models.training import ModelTrainer
from models.technical import compute_adx, compute_vwap_bands, prepare_features_for_training
from utils.logger import initialize_logging
from utils.symbols import read_symbols_from_file, clean_symbol


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train ML models for trading")
    parser.add_argument("--symbols", "-s", help="Path to symbols file")
    parser.add_argument("--days", "-d", type=int, default=DAYS_OF_HISTORICAL_DATA,
                        help=f"Days of history to fetch (default: {DAYS_OF_HISTORICAL_DATA})")
    parser.add_argument("--parallel", "-p", type=int, default=5,
                        help="Number of parallel workers (default: 5)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force retraining of existing models")
    parser.add_argument("--features", "-e", action="store_true",
                        help="Use enhanced features (ADX, VWAP, etc.)")
    parser.add_argument("--deep", action="store_true",
                        help="Train more complex model with more trees")
    parser.add_argument("--test-split", "-t", type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")
    return parser.parse_args()


def fetch_data_for_symbol(kite_client, symbol, days, output_dir, force=False):
    """
    Fetch historical data for a symbol.

    Args:
        kite_client: KiteClient instance
        symbol: Symbol to fetch data for
        days: Number of days of history to fetch
        output_dir: Directory to save data to
        force: Whether to force refetching of data

    Returns:
        tuple: (symbol, status, message)
    """
    try:
        # Clean symbol for consistent processing
        clean_sym = clean_symbol(symbol)
        output_file = os.path.join(output_dir, f"{clean_sym}_minute_data.csv")

        # Skip if data exists and not forcing
        if os.path.exists(output_file) and not force:
            file_size = os.path.getsize(output_file)
            if file_size > 1000:  # Non-empty file
                return symbol, "skip", f"Data file exists ({file_size / 1024:.1f} KB)"

        # Get instrument token
        token = kite_client.get_instrument_token("NSE", symbol)
        if not token:
            return symbol, "error", "Could not get instrument token"

        # Update historical data
        df = update_historical_data(kite_client, token, symbol, output_file, days=days)

        if df is not None and not df.empty:
            return symbol, "success", f"Fetched {len(df)} data points"
        else:
            return symbol, "error", "No data fetched"

    except Exception as e:
        return symbol, "error", f"Error: {str(e)}"


def enhanced_train_model_for_symbol(symbol, training_dir, pickles_dir, use_enhanced_features=True,
                                    deep_model=False, test_size=0.2, force=False):
    """
    Train an enhanced model for a symbol with additional features.

    Args:
        symbol: Symbol to train model for
        training_dir: Directory with training data
        pickles_dir: Directory to save models
        use_enhanced_features: Whether to use enhanced features
        deep_model: Whether to train a more complex model
        test_size: Test split ratio
        force: Whether to force retraining

    Returns:
        tuple: (symbol, status, message, metrics)
    """
    try:
        # Clean symbol for consistent processing
        clean_sym = clean_symbol(symbol)
        model_file = os.path.join(pickles_dir, f"{clean_sym}_scalping_model.pkl")

        # Check if model already exists
        if os.path.exists(model_file) and not force:
            return symbol, "skip", "Model already exists", None

        # Load data
        csv_path = os.path.join(training_dir, f"{clean_sym}_minute_data.csv")
        if not os.path.exists(csv_path):
            return symbol, "error", "No training data available", None

        # Read the CSV
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df.set_index("date", inplace=True)

        if df.empty:
            return symbol, "error", "Empty dataset", None

        # Prepare features
        if use_enhanced_features:
            # Add ADX
            df = compute_adx(df)

            # Add VWAP bands
            df = compute_vwap_bands(df)

            # Add momentum features
            df["momentum"] = df["close"] - df["close"].shift(4)
            df["momentum_acceleration"] = df["momentum"] - df["momentum"].shift(3)

            # Add volume features
            df["volume_ma"] = df["volume"].rolling(window=10, min_periods=3).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]

            # Add volatility
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(window=20, min_periods=5).std()

        # Use the standard feature preparation function
        df = prepare_features_for_training(df)

        if df is None or df.empty:
            return symbol, "error", "Failed to prepare features", None

        # Define enhanced feature set
        if use_enhanced_features:
            feature_cols = [
                "returns", "sma20", "ema20", "macd", "macd_signal", "rsi",
                "adx", "+di", "-di", "momentum", "momentum_acceleration", "volatility"
            ]
        else:
            feature_cols = ["returns", "sma20", "ema20", "macd", "macd_signal", "rsi"]

        # Keep only features that exist in the dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]

        if not feature_cols:
            return symbol, "error", "No usable features", None

        # Prepare for training
        X = df[feature_cols].values
        y = df["target"].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create model
        if deep_model:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        # Get predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        metrics = {
            "train_score": train_score,
            "test_score": test_score,
            "test_size": len(X_test),
            "feature_importance": dict(zip(feature_cols, model.feature_importances_))
        }

        # Now train on full dataset for final model
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        # Save model
        with open(model_file, "wb") as f:
            pickle.dump((model, scaler, feature_cols), f)

        return symbol, "success", f"Model trained with {len(feature_cols)} features, accuracy: {test_score:.2f}", metrics

    except Exception as e:
        return symbol, "error", f"Error: {str(e)}", None


def main():
    """Main execution function"""
    # Initialize logging
    logger = initialize_logging()
    logger.info("=" * 60)
    logger.info("STARTING ENHANCED MODEL TRAINING")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_arguments()

    # Load symbols
    symbols = read_symbols_from_file(args.symbols)
    if not symbols:
        logger.error("No symbols found in symbols file")
        return

    # Initialize components
    credentials = Credentials()
    creds = credentials.get_credentials()
    kite_client = KiteClient(creds['api_key'], creds['api_secret'])
    kite_client.initialize()

    # Ensure directories exist
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(PICKLES_DIR, exist_ok=True)

    # Step 1: Fetch historical data for all symbols
    logger.info(f"Step 1: Fetching historical data for {len(symbols)} symbols")

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit tasks
        future_to_symbol = {
            executor.submit(
                fetch_data_for_symbol,
                kite_client, symbol, args.days, TRAINING_DATA_DIR, args.force
            ): symbol for symbol in symbols
        }

        # Process results with progress bar
        successful_data = 0
        skipped_data = 0
        failed_data = 0

        with tqdm(total=len(future_to_symbol), desc="Fetching Data") as pbar:
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, status, message = future.result()

                    if status == "success":
                        successful_data += 1
                        logger.info(f"Data fetch for {symbol}: {message}")
                    elif status == "skip":
                        skipped_data += 1
                        logger.debug(f"Skipping data fetch for {symbol}: {message}")
                    else:
                        failed_data += 1
                        logger.error(f"Failed to fetch data for {symbol}: {message}")

                except Exception as e:
                    failed_data += 1
                    logger.error(f"Exception processing {symbol}: {e}")

                pbar.update(1)

    logger.info(f"Data fetch summary: {successful_data} successful, {skipped_data} skipped, {failed_data} failed")

    # Step 2: Train enhanced models
    logger.info(f"Step 2: Training {'enhanced' if args.features else 'standard'} models for {len(symbols)} symbols")

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit tasks
        future_to_symbol = {
            executor.submit(
                enhanced_train_model_for_symbol,
                symbol, TRAINING_DATA_DIR, PICKLES_DIR,
                args.features, args.deep, args.test_split, args.force
            ): symbol for symbol in symbols
        }

        # Process results with progress bar
        successful_models = 0
        skipped_models = 0
        failed_models = 0

        # Track metrics
        all_metrics = {}

        with tqdm(total=len(future_to_symbol), desc="Training Models") as pbar:
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, status, message, metrics = future.result()

                    if status == "success":
                        successful_models += 1
                        logger.info(f"Model training for {symbol}: {message}")
                        if metrics:
                            all_metrics[symbol] = metrics
                    elif status == "skip":
                        skipped_models += 1
                        logger.debug(f"Skipping model training for {symbol}: {message}")
                    else:
                        failed_models += 1
                        logger.error(f"Failed to train model for {symbol}: {message}")

                except Exception as e:
                    failed_models += 1
                    logger.error(f"Exception training model for {symbol}: {e}")

                pbar.update(1)

    logger.info(
        f"Model training summary: {successful_models} successful, {skipped_models} skipped, {failed_models} failed")

    # Report overall metrics
    if all_metrics:
        avg_train_score = np.mean([m["train_score"] for m in all_metrics.values()])
        avg_test_score = np.mean([m["test_score"] for m in all_metrics.values()])

        logger.info(f"Average training accuracy: {avg_train_score:.4f}")
        logger.info(f"Average testing accuracy: {avg_test_score:.4f}")

        # Find most important features across all models
        feature_importance = {}
        for symbol, metrics in all_metrics.items():
            if "feature_importance" in metrics:
                for feature, importance in metrics["feature_importance"].items():
                    if feature not in feature_importance:
                        feature_importance[feature] = []
                    feature_importance[feature].append(importance)

        # Calculate average importance
        avg_importance = {f: np.mean(imps) for f, imps in feature_importance.items()}

        # Get top 5 features
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        logger.info("Top 5 important features:")
        for feature, importance in top_features:
            logger.info(f"  {feature}: {importance:.4f}")

    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print(traceback.format_exc())