# Symbolic execution, hero style. :-)

import dis
from typing import Callable


def nope(arg):
    pass


def test(a, b):
    for i in range(3):
        nope(a*i + b)


test(2, 4)

dis.dis(test, adaptive=True)

"""
  6           0 RESUME                   0

  7           2 LOAD_GLOBAL              1 (NULL + range)
             14 LOAD_CONST               1 (10)
             16 CALL                     1
             26 GET_ITER
        >>   28 FOR_ITER_RANGE          21 (to 74)
             32 STORE_FAST               2 (i)

  8          34 LOAD_GLOBAL_BUILTIN      3 (NULL + nope)
             46 LOAD_FAST__LOAD_FAST     0 (a)
             48 LOAD_FAST                2 (i)
             50 BINARY_OP_MULTIPLY_INT     5 (*)
             54 LOAD_FAST                1 (b)
             56 BINARY_OP_ADD_INT        0 (+)
             60 CALL_BUILTIN_FAST_WITH_KEYWORDS     1
             70 POP_TOP
             72 JUMP_BACKWARD           23 (to 28)

  7     >>   74 END_FOR
             76 RETURN_CONST             0 (None)
"""


Stack = tuple[str, ...]

def update_stack(input: Stack, b: dis.Instruction) -> Stack:
    stack = list(input)
    baseopname = dis.deoptmap.get(b.opname, b.opname)
    opcode = dis.opmap[baseopname]
    if b.arg is not None:
        diff = dis.stack_effect(opcode, b.arg, jump=False)
    else:
        diff = dis.stack_effect(opcode, None, jump=False)
    if baseopname == "LOAD_GLOBAL" and diff == 2:
        stack.append("null")
        diff -= 1
    while diff > 0:
        stack.append("object")
        diff -= 1
    while diff < 0:
        stack.pop(-1)
        diff += 1
    if baseopname == "CALL":
        stack[-1] = "object"  # overwrite 'null'
    return tuple(stack)


def successors(b: dis.Instruction) -> None | tuple[bool, int]:
    baseopname = dis.deoptmap.get(b.opname, b.opname)
    opcode = dis.opmap[baseopname]
    assert not dis.hasjabs
    if opcode not in dis.hasjrel:
        return None
    fallthrough = True
    arg = b.arg
    if baseopname == "JUMP_BACKWARD":
        arg = -arg
        fallthrough = False
    return fallthrough, b.offset + 2*arg


def run(func: Callable[..., object]):
    instrs: list[dis.Instruction] = list(dis.Bytecode(func, adaptive=True))
    assert instrs != []
    stacks: dict[dis.Instruction, Stack | None] = {b: None for b in instrs}
    stacks[instrs[0]] = ()
    todo = True
    while todo:
        todo = False
        stack = None
        for b in instrs:
            if stack is not None:
                if stacks[b] is None:
                    stacks[b] = stack
                    todo = True
                else:
                    if stacks[b] != stack:
                        print(f"MISMATCH at {b.offset}: {stacks[b]} != {stack}")
            else:
                stack = stacks[b]
                if stack is None:
                    continue
            stack = update_stack(stack, b)
            jumps = successors(b)
            if jumps is not None:
                fallthrough, offset = jumps
                for bb in instrs:
                    if bb.offset == offset:
                        if stacks[bb] is None:
                            stacks[bb] = stack
                            todo = True
                        else:
                            if stacks[bb] != stack:
                                print(f"MISMATCH AT {offset}: {stacks[bb]} != {stack}")
                        break
                else:
                    print(f"CANNOT FIND {offset} for {stack}")
                if not fallthrough:
                    stack = None
        if stack is not None:
            print(f"End: {stack}")

    for b in instrs:
        stack = stacks.get(b)
        if stack is not None:
            stack = list(stack)
        print(str(stack).ljust(50), b.opname, b.arg if b.arg is not None else "")


run(test)
