#!/usr/bin/env python3
"""
Trading Bot Project Backup Script

Creates a zipped backup of the essential project files while excluding
virtual environments, generated data, logs, and other non-essential files.
"""

import os
import sys
import zipfile
import datetime
import shutil
import tempfile
from pathlib import Path

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Define backup name with timestamp
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_NAME = f"trading_bot_backup_{TIMESTAMP}.zip"

# Directories to include
INCLUDE_DIRS = [
    "api",
    "config",
    "models",
    "trading",
    "utils",
]

# Individual files to include
INCLUDE_FILES = [
    "main.py",
    "train.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Patterns to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".DS_Store",
    "*.log",
    "venv",
    "env",
    ".env",
    "logs",
    "pickles",
    "data/training",
    "config/credentials.enc",
]

# Make sure data/symbols directory is included but with only essential files
INCLUDE_SYMBOL_FILES = [
    "data/symbols/trading_symbols.txt",
]


def should_exclude(path):
    """Check if a path should be excluded based on patterns"""
    path_str = str(path)
    return any(pattern in path_str for pattern in EXCLUDE_PATTERNS)


def create_backup():
    """Create a zip backup of the project"""
    # Create a temporary directory to organize files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy directories
        for dir_name in INCLUDE_DIRS:
            source_dir = PROJECT_ROOT / dir_name
            if not source_dir.exists():
                print(f"Warning: Directory {dir_name} does not exist, skipping")
                continue

            dest_dir = temp_path / dir_name

            # Create directory structure
            os.makedirs(dest_dir, exist_ok=True)

            # Copy files while excluding patterns
            for root, dirs, files in os.walk(source_dir):
                rel_path = Path(root).relative_to(PROJECT_ROOT)

                # Skip if this directory should be excluded
                if should_exclude(rel_path):
                    continue

                # Remove directories that should be excluded
                dirs[:] = [d for d in dirs if not should_exclude(rel_path / d)]

                # Create the directory in temp
                os.makedirs(temp_path / rel_path, exist_ok=True)

                # Copy files that shouldn't be excluded
                for file in files:
                    if not should_exclude(rel_path / file):
                        shutil.copy2(
                            PROJECT_ROOT / rel_path / file,
                            temp_path / rel_path / file
                        )

        # Copy individual files
        for file_name in INCLUDE_FILES:
            source_file = PROJECT_ROOT / file_name
            if not source_file.exists():
                print(f"Warning: File {file_name} does not exist, skipping")
                continue

            shutil.copy2(source_file, temp_path / file_name)

        # Create data directory and add symbol files
        os.makedirs(temp_path / "data" / "symbols", exist_ok=True)

        for symbol_file in INCLUDE_SYMBOL_FILES:
            source_file = PROJECT_ROOT / symbol_file
            if not source_file.exists():
                print(f"Warning: Symbol file {symbol_file} does not exist, skipping")
                continue

            dest_file = temp_path / symbol_file
            os.makedirs(dest_file.parent, exist_ok=True)
            shutil.copy2(source_file, dest_file)

        # Create necessary empty directories to maintain structure
        empty_dirs = ["data/training", "logs", "pickles"]
        for empty_dir in empty_dirs:
            os.makedirs(temp_path / empty_dir, exist_ok=True)
            # Add .gitkeep file to ensure directory is included in git
            with open(temp_path / empty_dir / ".gitkeep", "w") as f:
                f.write("# This directory will contain generated files\n")

        # Create the zip file
        with zipfile.ZipFile(BACKUP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_path):
                rel_path = Path(root).relative_to(temp_path)

                # Write files
                for file in files:
                    file_path = Path(root) / file
                    archive_path = "trading_bot" / rel_path / file if rel_path != "." else "trading_bot" / file
                    zipf.write(file_path, archive_path)

    print(f"Backup created: {BACKUP_NAME}")
    print(f"Size: {os.path.getsize(BACKUP_NAME) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    try:
        create_backup()
    except Exception as e:
        print(f"Error creating backup: {e}")
        sys.exit(1)