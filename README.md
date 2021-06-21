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
  In addition, stats for a representative mypy run are also given.

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

Most common opcodes and pairs for 100 most popular PyPI packages
----------------------------------------------------------------

(Actuall 99 out of 100, since Windows Defender blocks `Pillow-8.2.0.tar.gz`.)

```
Total: 239 errors; 9,992 files; 218,124 code objects; 3,369,978 lines;
10,540,908 opcodes; 10,322,784 opcode pairs;
5,661,062.0 cache_size; 1,086,258.0 cache wasted;
 2,287,402 ops quickened; 876 skip quicken; 61,018 prev extended args

Top 20 single opcodes:
LOAD_CONST            2,258,158  21.42%
LOAD_FAST             1,371,243  13.01%
LOAD_GLOBAL             666,488   6.32%
LOAD_ATTR               561,987   5.33%
CALL_METHOD             474,246   4.50%
LOAD_METHOD             474,198   4.50%
STORE_FAST              457,434   4.34%
POP_TOP                 447,568   4.25%
CALL_FUNCTION           422,514   4.01%
STORE_NAME              386,370   3.67%
LOAD_NAME               267,183   2.53%
RETURN_VALUE            260,108   2.47%
MAKE_FUNCTION           208,166   1.97%
BUILD_LIST              175,689   1.67%
CALL_FUNCTION_KW        151,275   1.44%
POP_JUMP_IF_FALSE       116,176   1.10%
BINARY_SUBSCR           106,368   1.01%
BUILD_TUPLE             101,091   0.96%
POP_JUMP_IF_TRUE         99,945   0.95%
COMPARE_OP               91,309   0.87%

Top 20 opcode pairs:
LOAD_CONST           --> LOAD_CONST              675,323   6.54%
LOAD_FAST            --> LOAD_ATTR               276,461   2.68%
STORE_NAME           --> LOAD_CONST              223,836   2.17%
LOAD_FAST            --> LOAD_FAST               212,996   2.06%
LOAD_CONST           --> MAKE_FUNCTION           208,166   2.02%
LOAD_FAST            --> LOAD_CONST              196,262   1.90%
LOAD_FAST            --> LOAD_METHOD             193,238   1.87%
STORE_FAST           --> LOAD_FAST               187,288   1.81%
LOAD_GLOBAL          --> LOAD_FAST               178,732   1.73%
LOAD_CONST           --> RETURN_VALUE            172,984   1.68%
LOAD_METHOD          --> LOAD_FAST               166,124   1.61%
STORE_FAST           --> LOAD_GLOBAL             152,105   1.47%
LOAD_CONST           --> CALL_FUNCTION_KW        151,275   1.47%
CALL_METHOD          --> POP_TOP                 146,864   1.42%
LOAD_GLOBAL          --> LOAD_METHOD             134,456   1.30%
LOAD_METHOD          --> LOAD_CONST              130,992   1.27%
MAKE_FUNCTION        --> STORE_NAME              127,850   1.24%
POP_TOP              --> LOAD_CONST              124,477   1.21%
LOAD_ATTR            --> LOAD_CONST              124,227   1.20%
LOAD_GLOBAL          --> LOAD_ATTR               117,403   1.14%
```

Most common opcodes and pairs for mypy
--------------------------------------

This is useful for comparison, since mypy is a very different code base.

```
Total: 164 files; 6,010 code objects; 76,147 lines;
299,904 opcodes; 293,894 opcode pairs;
161,380.0 cache_size; 33,656.0 cache wasted;
63,862 ops quickened; 2,309 prev extended args

Top 20 single opcodes:
LOAD_CONST               51,195  17.07%
LOAD_FAST                49,104  16.37%
LOAD_ATTR                20,759   6.92%
LOAD_GLOBAL              17,927   5.98%
LOAD_NAME                14,527   4.84%
CALL_FUNCTION            12,555   4.19%
STORE_NAME               11,578   3.86%
STORE_FAST               11,049   3.68%
CALL_METHOD              10,828   3.61%
LOAD_METHOD              10,823   3.61%
RETURN_VALUE              9,028   3.01%
POP_TOP                   8,959   2.99%
POP_JUMP_IF_FALSE         7,910   2.64%
BUILD_TUPLE               6,769   2.26%
MAKE_FUNCTION             5,848   1.95%
BINARY_SUBSCR             5,256   1.75%
IMPORT_FROM               4,034   1.35%
BUILD_LIST                2,908   0.97%
POP_JUMP_IF_TRUE          2,572   0.86%
JUMP_ABSOLUTE             2,554   0.85%

Top 20 opcode pairs:
LOAD_FAST            --> LOAD_ATTR                14,505   4.94%
LOAD_CONST           --> LOAD_CONST               12,495   4.25%
LOAD_CONST           --> LOAD_NAME                 9,437   3.21%
LOAD_GLOBAL          --> LOAD_FAST                 7,848   2.67%
LOAD_FAST            --> LOAD_FAST                 6,370   2.17%
LOAD_METHOD          --> LOAD_FAST                 6,232   2.12%
LOAD_FAST            --> LOAD_METHOD               6,007   2.04%
STORE_NAME           --> LOAD_CONST                5,937   2.02%
LOAD_CONST           --> MAKE_FUNCTION             5,848   1.99%
LOAD_NAME            --> LOAD_CONST                5,090   1.73%
STORE_FAST           --> LOAD_FAST                 5,014   1.71%
LOAD_CONST           --> RETURN_VALUE              4,605   1.57%
BUILD_TUPLE          --> LOAD_CONST                4,406   1.50%
POP_JUMP_IF_FALSE    --> LOAD_FAST                 4,271   1.45%
LOAD_FAST            --> CALL_METHOD               4,260   1.45%
CALL_METHOD          --> POP_TOP                   4,234   1.44%
MAKE_FUNCTION        --> STORE_NAME                4,019   1.37%
IMPORT_FROM          --> STORE_NAME                3,994   1.36%
LOAD_NAME            --> LOAD_NAME                 3,614   1.23%
LOAD_ATTR            --> LOAD_FAST                 3,579   1.22%
```

Dynamic stats for pyperformance benchmarks
------------------------------------------

For the full data, see [dxpstats.txt](./dxpstats.txt).
The average of the percentages there comes down to this:
```
LOAD_FAST            --> LOAD_FAST              6.08%
STORE_FAST           --> LOAD_FAST              4.47%
LOAD_FAST            --> LOAD_ATTR              4.08%
LOAD_FAST            --> LOAD_CONST             3.21%
POP_JUMP_IF_FALSE    --> LOAD_FAST              3.07%
JUMP_ABSOLUTE        --> FOR_ITER               2.08%
LOAD_GLOBAL          --> LOAD_FAST              2.02%
LOAD_FAST            --> LOAD_METHOD            1.80%
FOR_ITER             --> STORE_FAST             1.66%
LOAD_ATTR            --> LOAD_FAST              1.62%
COMPARE_OP           --> POP_JUMP_IF_FALSE      1.23%
LOAD_FAST            --> LOAD_GLOBAL            1.21%
LOAD_FAST            --> CALL_FUNCTION          1.17%
LOAD_METHOD          --> LOAD_FAST              1.11%
LOAD_FAST            --> BINARY_MULTIPLY        0.91%
LOAD_FAST            --> STORE_ATTR             0.90%
STORE_FAST           --> STORE_FAST             0.89%
IS_OP                --> POP_JUMP_IF_FALSE      0.84%
LOAD_CONST           --> LOAD_FAST              0.75%
LOAD_FAST            --> CALL_METHOD            0.74%
```

Note that this is the average of the top 20 entries per program,
which causes some rounding down (I didn't have the full data set handy).

Also note that this averages the *percentages*, not the total counts.
This is because different benchmarks have a different overall opcode count;
it would seem wrong if a benchmark that happens to run more loops
contributes proportionally more to these statistics.
