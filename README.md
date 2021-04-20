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

```
Total: 119 errors; 9,219 files; 67,289 code objects; 3,163,537 lines; 4,187,645 opcodes; 4,120,356 opcode pairs

Top 20 single opcodes:
LOAD_CONST            1,144,736  27.34%
STORE_NAME              348,148   8.31%
LOAD_FAST               335,094   8.00%
LOAD_NAME               213,438   5.10%
LOAD_GLOBAL             188,964   4.51%
MAKE_FUNCTION           174,962   4.18%
CALL_FUNCTION           173,139   4.13%
LOAD_ATTR               150,468   3.59%
POP_TOP                 148,246   3.54%
CALL_METHOD             131,922   3.15%
LOAD_METHOD             131,914   3.15%
STORE_FAST              131,170   3.13%
RETURN_VALUE             82,200   1.96%
IMPORT_FROM              64,972   1.55%
BUILD_LIST               62,071   1.48%
IMPORT_NAME              60,974   1.46%
BUILD_TUPLE              46,516   1.11%
CALL_FUNCTION_KW         44,033   1.05%
POP_JUMP_IF_FALSE        40,801   0.97%
POP_JUMP_IF_TRUE         35,322   0.84%

Top 20 opcode pairs:
LOAD_CONST           --> LOAD_CONST              429,298  10.42%
STORE_NAME           --> LOAD_CONST              201,837   4.90%
LOAD_CONST           --> MAKE_FUNCTION           174,962   4.25%
MAKE_FUNCTION        --> STORE_NAME              119,871   2.91%
IMPORT_FROM          --> STORE_NAME               62,950   1.53%
LOAD_CONST           --> IMPORT_NAME              60,973   1.48%
LOAD_FAST            --> LOAD_FAST                60,922   1.48%
LOAD_FAST            --> LOAD_CONST               59,220   1.44%
LOAD_CONST           --> RETURN_VALUE             58,266   1.41%
LOAD_NAME            --> LOAD_ATTR                57,783   1.40%
POP_TOP              --> LOAD_CONST               53,957   1.31%
LOAD_GLOBAL          --> LOAD_FAST                52,899   1.28%
STORE_FAST           --> LOAD_FAST                51,620   1.25%
LOAD_CONST           --> LOAD_NAME                50,608   1.23%
LOAD_CONST           --> STORE_NAME               48,762   1.18%
CALL_FUNCTION        --> STORE_NAME               48,395   1.17%
STORE_NAME           --> LOAD_NAME                47,332   1.15%
LOAD_METHOD          --> LOAD_CONST               46,528   1.13%
STORE_FAST           --> LOAD_GLOBAL              44,870   1.09%
LOAD_CONST           --> CALL_FUNCTION_KW         44,033   1.07%
```

Most common opcodes and pairs for mypy
--------------------------------------

This is useful for comparison, since mypy is a very different code base.

```
Total: 160 files; 1,396 code objects; 74,668 lines; 108,133 opcodes; 106,737 opcode pairs

Top 20 single opcodes:
LOAD_CONST               26,507  24.51%
LOAD_FAST                11,412  10.55%
STORE_NAME               11,359  10.50%
LOAD_GLOBAL               5,589   5.17%
MAKE_FUNCTION             5,162   4.77%
CALL_FUNCTION             4,515   4.18%
LOAD_ATTR                 4,431   4.10%
IMPORT_FROM               3,955   3.66%
STORE_FAST                3,294   3.05%
POP_TOP                   3,122   2.89%
EXTENDED_ARG              2,668   2.47%
POP_JUMP_IF_FALSE         2,418   2.24%
CALL_METHOD               2,379   2.20%
LOAD_METHOD               2,374   2.20%
RETURN_VALUE              2,351   2.17%
LOAD_NAME                 2,098   1.94%
IMPORT_NAME               1,368   1.27%
BUILD_TUPLE                 917   0.85%
BINARY_SUBSCR               874   0.81%
LOAD_DEREF                  733   0.68%

Top 20 opcode pairs:
LOAD_CONST           --> LOAD_CONST               11,265  10.55%
LOAD_CONST           --> MAKE_FUNCTION             5,162   4.84%
STORE_NAME           --> LOAD_CONST                4,886   4.58%
MAKE_FUNCTION        --> STORE_NAME                4,040   3.79%
IMPORT_FROM          --> STORE_NAME                3,930   3.68%
STORE_NAME           --> IMPORT_FROM               2,898   2.72%
LOAD_FAST            --> LOAD_ATTR                 2,753   2.58%
LOAD_GLOBAL          --> LOAD_FAST                 2,534   2.37%
EXTENDED_ARG         --> LOAD_CONST                2,267   2.12%
LOAD_FAST            --> LOAD_FAST                 1,683   1.58%
STORE_FAST           --> LOAD_FAST                 1,426   1.34%
LOAD_CONST           --> STORE_NAME                1,395   1.31%
LOAD_CONST           --> IMPORT_NAME               1,368   1.28%
POP_TOP              --> LOAD_CONST                1,203   1.13%
LOAD_FAST            --> CALL_FUNCTION             1,097   1.03%
POP_JUMP_IF_FALSE    --> LOAD_FAST                 1,074   1.01%
LOAD_CONST           --> EXTENDED_ARG              1,061   0.99%
IMPORT_NAME          --> IMPORT_FROM               1,054   0.99%
LOAD_FAST            --> LOAD_CONST                1,045   0.98%
LOAD_FAST            --> LOAD_METHOD               1,040   0.97%
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
