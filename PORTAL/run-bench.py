import os.path
import subprocess
import sys


proc = subprocess.run(
    [sys.executable,
     os.path.join(os.path.dirname(__file__), 'jobs.py'),
     # XXX Change the flag to --run-attached once that works.
     'add', 'compile-bench', '--no-run',
     *sys.argv[1:],
    ]
)
sys.exit(proc.returncode)
