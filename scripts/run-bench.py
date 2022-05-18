#! /usr/bin/python3.8

import os.path
import subprocess
import sys


proc = subprocess.run(
    [sys.executable, '-u', '-m', 'jobs',
     # XXX Change the flag to --run-attached once that works.
     'add', 'compile-bench', '--run-attached',
     *sys.argv[1:],
    ],
    cwd=os.path.join(os.path.dirname(__file__)),
)
sys.exit(proc.returncode)
