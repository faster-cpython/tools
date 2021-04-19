Tools for gathering bytecode statistics
=======================================

Here are tools for gathering *dynamic* and *static* statistics about
bytecode frequencey and the like.

- Dynamic statistics run a program with a Python recompiled with
  `-DDYNAMIC_EXECUTION_PROFILE -DDXPAIRS` and at the end call
  `sys.getdxp()`, which returns an array of counters that is dumped
  to a json file. This can them be analyzed further by
  `scripts/analyze_dxp.py`.
  Because gathering this data is pretty tedious,
  the repo contains the raw JSON data for many of the "pyperformance"
  benchmarks, gathered using `scripts/dxp_pyperformance.py`.

- Static statistics are easier to collect, since no execution of the
  code is required, and the runtime dependencies are not needed.
  However, we don't know how well these static statistics predict
  the dynamic statistics (and the latter are what we want to improve).
  To download the 100 most popular PyPI packages, run
  `scripts/download_packages.py`; this places files in `packages`.
  To analyze packages or modules, run `scripts/count_opcodes.py`,
  pointing it to one or more `.py` source code files,
  directories full of those, or `.tar.gz` archives.
