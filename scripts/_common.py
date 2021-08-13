import os.path
import sys


SCRIPTS_DIR =  os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
LIB_DIR = os.path.join(ROOT_DIR, 'lib')

# Make fc_utils, fc_tools, etc. available.
sys.path.insert(0, LIB_DIR)
