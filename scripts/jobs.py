#! /usr/bin/python3.8

import os.path
import subprocess
import sys

if __name__ != '__main__':
    sys.exit('not a module')

SCRIPTS_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
TOOLS_ROOT = os.path.dirname(SCRIPTS_DIR)

proc = subprocess.run(
    [sys.executable, '-u', '-m', 'jobs', *sys.argv[1:]],
    cwd=os.path.join(TOOLS_ROOT, 'PORTAL'),
)
sys.exit(proc.returncode)

