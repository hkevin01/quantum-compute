"""
Unit tests for quantum computing research projects.
"""

import os
import sys
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tests.test_cosmology import *
from tests.test_medical import *
from tests.test_ml import *
from tests.test_utils import *

if __name__ == '__main__':
    unittest.main()
