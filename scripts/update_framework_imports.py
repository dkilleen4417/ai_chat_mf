"""
Script to update framework files to use the new api_keys utility.

This script updates all framework files in the src/frameworks directory to use
the new api_keys utility for better API key management.
"""

import os
import re
from pathlib import Path

# Directory containing the framework files
FRAMEWORKS_DIR = Path("src/frameworks")

# Files to update (relative to FRAMEWORKS_DIR)
FRAMEWORK_FILES = [
    "anthropic.py",
    "chatgpt.py",
    "groq.py",
    "grok.py",
    "llama.py",
    "ollama.py"
]

# Template for the import statement
IMPORT_STATEMENT = (
    "import streamlit as st\n"
    "from typing import Dict, List, Optional\n"
    "\n"
    "# Import API key utilities\n"
    "from utils.api_keys import get_api_key, validate_api_key\n\n"
)

# Pattern to match the existing import section
IMPORT_PATTERN = r'(?:^#.*\n)*import .*\n(?:from .*\n)*\n'

# Pattern to match API key retrieval
API_KEY_PATTERN = r'api_key\s*=\s*st\.secrets\.get\(([^)]+)\)'

# Replacement for API key retrieval
API_KEY_REPLACEMENT = (
    '    # Safely get and validate the API key\n'
    '    api_key = get_api_key(\1)\n'
    '    is_valid, error_msg = validate_api_key(api_key, "API key")\n'
    '    if not is_valid:\n'
    '        return {\'error\': error_msg}\n'
)

def update_file(file_path: Path) -> bool:
    """Update a single framework file with the new imports and API key handling."""
    try:
        content = file_path.read_text()
        
        # Skip if already updated
        if 'from utils.api_keys import' in content:
            print(f"Skipping {file_path} - already updated")
            return False
            
        # Add import statement
        content = re.sub(
            IMPORT_PATTERN,
            IMPORT_STATEMENT,
            content,
            count=1,
            flags=re.MULTILINE
        )
        
        # Update API key retrieval
        content = re.sub(
            API_KEY_PATTERN,
            API_KEY_REPLACEMENT,
            content,
            count=1,
            flags=re.MULTILINE
        )
        
        # Write the updated content back to the file
        file_path.write_text(content)
        print(f"Updated {file_path}")
        return True
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all framework files."""
    updated = 0
    skipped = 0
    errors = 0
    
    for filename in FRAMEWORK_FILES:
        file_path = FRAMEWORKS_DIR / filename
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist")
            errors += 1
            continue
            
        if update_file(file_path):
            updated += 1
        else:
            skipped += 1
    
    print(f"\nUpdate complete!")
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()
