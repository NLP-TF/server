import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up test environment variables
os.environ["ENV"] = "test"

# Import fixtures here if needed
# from app.tests.fixtures import *
