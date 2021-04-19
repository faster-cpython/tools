# Run Python with dxpairs over selected benchmarks.
#
# Instructions to run:
# - Check out CPython, add these #defines to Include/internal/pycore_ceval.h:
#   #define DYNAMIC_EXECUTION_PROFILE
#   #define DXPAIRS
# - Build and install CPython; use it for everything below
#
# - Check out pyperformance as ~/pyperformance
# - On Windows, modify pyperformace requirements.{txt,in} to
#   comment out SQLAlchemy and greenlet
# - pip-install the modified pyperformance
# - Run pyperformance pyperformance venv
#   (this will pip-install a bunch of stuff)
#
# - Take the venv printed by that command and paste it below for VENV
# - Run this script (scripts/dxp_pyperformance.py)
#  (to debug, run with a single benchmark name, e.g. 'nqueens')
#
# - Then run scripts/analyze_dxp.py data/*.json with various args
#   to analyze the data

VENV = "~/pyperformance/venv/cpython3.10-ca66253da885/Lib/site-packages"
BENCHDIR = "~/pyperformance/pyperformance/benchmarks"

BENCHMARKS = """
- 2to3
- chameleon
- chaos
- crypto_pyaes
- deltablue
- django_template
- dulwich_log
- fannkuch
- float
- genshi
- go
- hexiom
- json_dumps
- json_loads
- logging
- mako
- meteor_contest
- nbody
- nqueens
- pathlib
- pidigits
- pyflate
- python_startup
- raytrace
- regex_compile
- regex_dna
- regex_effbot
- regex_v8
- richards
- scimark
- spectral_norm
- sqlite_synth
- sympy
- telco
- tornado_http
- unpack_sequence
- xml_etree
""".replace("-", "").split()

# These have complex command lines or otherwise don't run for me
SKIPPED_BM = """
- pickle
- pickle_dict
- pickle_list
- pickle_pure_python
- python_startup_no_site
- sqlalchemy_declarative
- sqlalchemy_imperative
- unpickle
- unpickle_list
- unpickle_pure_python
"""

import json
import os
import subprocess
import sys


def run_bm(name):
    if not hasattr(sys, "getdxp"):
        sys.exit("This Python is not compiled with -DDYNAMIC_EXECUTION_PAIRS")
    # Modify sys.path so we can find the benchmarks and their dependencies
    venv = os.path.expanduser(VENV)
    print(venv)
    sys.path.insert(0, venv)
    benchdir = os.path.expanduser(BENCHDIR)
    sys.path.append(benchdir)

    if not name.startswith("bm_"):
        name = "bm_" + name
    print("Running", name)
    sys.argv[1:] = ["--worker", "--loop=1", "--warmups=0"]
    filename = os.path.join(benchdir, name + ".py")
    with open(filename, "rb") as f:
        source = f.read()
    code = compile(source, filename, "exec")
    ns = {"__name__": "__main__", "__file__": filename}
    sys.getdxp()  # Flush earlier data
    exec(code, ns)
    dxp = sys.getdxp()
    data = json.dumps(dxp)
    os.makedirs("data", exist_ok=True)
    datafile = os.path.join("data", name[3:] + ".json")
    with open(datafile, "w") as f:
        f.write(data + "\n")
    print("JSON data written to", datafile)


def run_bm_in_subprocess(name):
    print()
    subprocess.check_call([sys.executable, sys.argv[0], name])


def main():
    names = sys.argv[1:]
    if not names:
        names = BENCHMARKS
    if len(names) == 1:
        run_bm(names[0])
    else:
        for name in names:
            run_bm_in_subprocess(name)


if __name__ == "__main__":
    main()
