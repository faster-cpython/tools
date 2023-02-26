"""Sample code for symex.py."""


def nope(arg):
    pass


def test(a, b):
    try:
        for i in range(3):
            nope(a * i + b / (2 - i))
    except Exception as err:
        nope(err)


test(1, 2)


# Output from `dis.dis(test, adaptive=True)`.

"""
  8           0 RESUME                   0

  9           2 NOP

 10           4 LOAD_GLOBAL              1 (NULL + range)
             16 LOAD_CONST               1 (3)
             18 CALL                     1
             28 GET_ITER
        >>   30 FOR_ITER_RANGE          27 (to 88)
             34 STORE_FAST               2 (i)

 11          36 LOAD_GLOBAL_MODULE       3 (NULL + nope)
             48 LOAD_FAST__LOAD_FAST     0 (a)
             50 LOAD_FAST                2 (i)
             52 BINARY_OP_MULTIPLY_INT     5 (*)
             56 LOAD_FAST__LOAD_CONST     1 (b)
             58 LOAD_CONST__LOAD_FAST     2 (2)
             60 LOAD_FAST                2 (i)
             62 BINARY_OP_SUBTRACT_INT    10 (-)
             66 BINARY_OP               11 (/)
             70 BINARY_OP                0 (+)
             74 CALL_PY_EXACT_ARGS       1
             84 POP_TOP
             86 JUMP_BACKWARD           29 (to 30)

 10     >>   88 END_FOR
             90 RETURN_CONST             0 (None)
        >>   92 PUSH_EXC_INFO

 12          94 LOAD_GLOBAL              4 (Exception)
            106 CHECK_EXC_MATCH
            108 POP_JUMP_IF_FALSE       23 (to 156)
            110 STORE_FAST               3 (err)

 13         112 LOAD_GLOBAL              3 (NULL + nope)
            124 LOAD_FAST                3 (err)
            126 CALL                     1
            136 POP_TOP
            138 POP_EXCEPT
            140 LOAD_CONST               0 (None)
            142 STORE_FAST               3 (err)
            144 DELETE_FAST              3 (err)
            146 RETURN_CONST             0 (None)
        >>  148 LOAD_CONST               0 (None)
            150 STORE_FAST               3 (err)
            152 DELETE_FAST              3 (err)
            154 RERAISE                  1

 12     >>  156 RERAISE                  0
        >>  158 COPY                     3
            160 POP_EXCEPT
            162 RERAISE                  1
ExceptionTable:
  4 to 88 -> 92 [0]
  92 to 110 -> 158 [1] lasti
  112 to 136 -> 148 [1] lasti
  148 to 156 -> 158 [1] lasti
"""
