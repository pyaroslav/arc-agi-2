"""Put ``src`` on sys.path so tests can ``import arc`` / ``import eval``."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
