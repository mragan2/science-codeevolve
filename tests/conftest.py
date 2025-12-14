import os
import sys

# Ensure the src directory is on the Python path for test discovery
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
