"""
Adds src/ to sys.path so tests can `import qr_solver` without the
project needing a package (__init__.py) layout yet.
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
