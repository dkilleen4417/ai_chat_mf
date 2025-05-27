#!/usr/bin/env python3
"""
Source Code Backup Script

This script creates a timestamped backup of the project's source code.
It creates a compressed tarball of the specified directories.
"""

import os
import tarfile
import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backup_source.log')
    ]
)
logger = logging.getLogger(__name__)

def create_source_backup(project_root, output_dir='./backups', exclude_dirs=None):
    """
    Create a compressed backup of the source code.
    
    Args:
        project_root (str): Root directory of the project
        output_dir (str): Directory to store the backup
        exclude_dirs (list): List of directory names to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', 'venv', 'node_modules', '.idea', '.vscode', 'backups']
    
    try:
        project_root = Path(project_root).resolve()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{project_root.name}_src_{timestamp}.tar.gz"
        backup_path = Path(output_dir) / backup_name
        
        # Create output directory if it doesn't exist
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating source code backup: {backup_path}")
        
        with tarfile.open(backup_path, 'w:gz') as tar:
            # Add project files, excluding specified directories
            for item in project_root.glob('*'):
                if item.name in exclude_dirs or item.name.startswith('.'):
                    continue
                logger.debug(f"Adding to backup: {item}")
                tar.add(item, arcname=item.name, recursive=True)
        
        logger.info(f"Successfully created source backup at {backup_path}")
        return True, str(backup_path)
        
    except Exception as e:
        logger.exception("An error occurred during source backup")
        return False, str(e)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup project source code')
    parser.add_argument('--project-root', default='..', 
                       help='Root directory of the project (default: parent of scripts dir)')
    parser.add_argument('--output-dir', default='../backups', 
                       help='Output directory for backups (default: ../backups)')
    
    args = parser.parse_args()
    
    success, message = create_source_backup(
        args.project_root,
        args.output_dir
    )
    
    if not success:
        logger.error(f"Backup failed: {message}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
