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

Popular stats
=============

Most common opcodes and pairs over 100 most popular PyPI packages
-----------------------------------------------------------------
```
Total: 119 errors; 9,219 files; 67,289 code objects; 3,163,537 lines; 4,187,645 opcodes; 4,120,356 opcode pairs

Top 20 single opcodes:
LOAD_CONST: 1,144,736 (27.34%)
STORE_NAME: 348,148 (8.31%)
LOAD_FAST: 335,094 (8.00%)
LOAD_NAME: 213,438 (5.10%)
LOAD_GLOBAL: 188,964 (4.51%)
MAKE_FUNCTION: 174,962 (4.18%)
CALL_FUNCTION: 173,139 (4.13%)
LOAD_ATTR: 150,468 (3.59%)
POP_TOP: 148,246 (3.54%)
CALL_METHOD: 131,922 (3.15%)
LOAD_METHOD: 131,914 (3.15%)
STORE_FAST: 131,170 (3.13%)
RETURN_VALUE: 82,200 (1.96%)
IMPORT_FROM: 64,972 (1.55%)
BUILD_LIST: 62,071 (1.48%)
IMPORT_NAME: 60,974 (1.46%)
BUILD_TUPLE: 46,516 (1.11%)
CALL_FUNCTION_KW: 44,033 (1.05%)
POP_JUMP_IF_FALSE: 40,801 (0.97%)
POP_JUMP_IF_TRUE: 35,322 (0.84%)

Top 20 opcode pairs:
LOAD_CONST => LOAD_CONST: 429,298 (10.42%)
STORE_NAME => LOAD_CONST: 201,837 (4.90%)
LOAD_CONST => MAKE_FUNCTION: 174,962 (4.25%)
MAKE_FUNCTION => STORE_NAME: 119,871 (2.91%)
IMPORT_FROM => STORE_NAME: 62,950 (1.53%)
LOAD_CONST => IMPORT_NAME: 60,973 (1.48%)
LOAD_FAST => LOAD_FAST: 60,922 (1.48%)
LOAD_FAST => LOAD_CONST: 59,220 (1.44%)
LOAD_CONST => RETURN_VALUE: 58,266 (1.41%)
LOAD_NAME => LOAD_ATTR: 57,783 (1.40%)
POP_TOP => LOAD_CONST: 53,957 (1.31%)
LOAD_GLOBAL => LOAD_FAST: 52,899 (1.28%)
STORE_FAST => LOAD_FAST: 51,620 (1.25%)
LOAD_CONST => LOAD_NAME: 50,608 (1.23%)
LOAD_CONST => STORE_NAME: 48,762 (1.18%)
CALL_FUNCTION => STORE_NAME: 48,395 (1.17%)
STORE_NAME => LOAD_NAME: 47,332 (1.15%)
LOAD_METHOD => LOAD_CONST: 46,528 (1.13%)
STORE_FAST => LOAD_GLOBAL: 44,870 (1.09%)
LOAD_CONST => CALL_FUNCTION_KW: 44,033 (1.07%)
```

Dynamic stats
-------------

Coming soon.
