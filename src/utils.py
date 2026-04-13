import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import METADATA_PATH


def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: No metadata file found at {METADATA_PATH}. Run ingest.py first.")
        return {}