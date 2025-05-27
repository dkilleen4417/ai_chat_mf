#!/usr/bin/env python3
"""
MongoDB Backup Script

This script creates a backup of a MongoDB database using mongodump.
It reads the database configuration from Streamlit's secrets and creates
timestamped backups in a specified directory.

Usage:
    python backup_mongodb.py [--out <output_dir>]

Example:
    python backup_mongodb.py --out ./backups
"""

import os
import subprocess
import argparse
import datetime
import logging
import sys
from pathlib import Path

# Add project root to path to import app
sys.path.append(str(Path(__file__).parent.parent))

# Import Streamlit to access secrets
import streamlit as st

# Load Streamlit secrets
st.secrets.load_if_toml_exists()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backup_mongodb.log')
    ]
)
logger = logging.getLogger(__name__)

def create_backup(output_dir):
    """
    Create a backup of the MongoDB database specified in Streamlit secrets.
    
    Args:
        output_dir (str): Directory to store the backup
    """
    try:
        # Get MongoDB configuration from Streamlit secrets
        mongodb_uri = st.secrets.get("MONGODB_URL", "mongodb://localhost:27017")
        db_name = st.secrets.get("MONGODB_DB_NAME", "chat_mf")
        
        # Create output directory if it doesn't exist
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path(output_dir) / f"{db_name}_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting backup of database '{db_name}' to {backup_dir}")
        
        # Build the mongodump command
        cmd = [
            'mongodump',
            f'--uri={mongodb_uri}',
            f'--db={db_name}',
            f'--out={backup_dir}'
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully created backup at {backup_dir}")
            return True, str(backup_dir)
        else:
            logger.error(f"Backup failed with error: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        logger.exception("An error occurred during backup")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Backup MongoDB database using Streamlit secrets')
    parser.add_argument('--out', default='./backups', 
                       help='Output directory for backups (default: ./backups)')
    
    args = parser.parse_args()
    
    try:
        success, message = create_backup(args.out)
        if not success:
            logger.error(f"Backup failed: {message}")
            return 1
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
