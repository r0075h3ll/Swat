import logging
import os
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
