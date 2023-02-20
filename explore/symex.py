#!/usr/bin/env python3.12

"""Symbolic execution, hero style. :-)"""

import dis
import importlib
import os
import sys
import types
from typing import Any, Callable, Iterable

assert not dis.hasjabs

import opcode_metadata as opcode_metadata_module

opcode_metadata_module.FVS_MASK = 0x4
opcode_metadata_module.FVS_HAVE_SPEC = 0x4

opcode_metadata: dict[str, dict[str, Any]]
from opcode_metadata import opcode_metadata  # Generated by cases_generator.py

opcode_metadata["COMPARE_AND_BRANCH"]["pushed"] = lambda oparg, jump: 1


def nope(arg):
    pass


def test(a, b):
    try:
        for i in range(3):
            nope(a * i + b / (2 - i))
    except Exception as err:
        nope(err)


"""
 17           0 RESUME                   0

 18           2 NOP

 19           4 LOAD_GLOBAL              1 (NULL + range)
             16 LOAD_CONST               1 (3)
             18 CALL                     1
             28 GET_ITER
        >>   30 FOR_ITER_RANGE          27 (to 88)
             34 STORE_FAST               2 (i)

 20          36 LOAD_GLOBAL_MODULE       3 (NULL + nope)
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

 19     >>   88 END_FOR
             90 LOAD_CONST               0 (None)
             92 RETURN_VALUE
        >>   94 PUSH_EXC_INFO

 21          96 LOAD_GLOBAL              4 (Exception)
            108 CHECK_EXC_MATCH
            110 POP_JUMP_IF_FALSE       24 (to 160)
            112 STORE_FAST               3 (err)

 22         114 LOAD_GLOBAL              3 (NULL + nope)
            126 LOAD_FAST                3 (err)
            128 CALL                     1
            138 POP_TOP
            140 POP_EXCEPT
            142 LOAD_CONST               0 (None)
            144 STORE_FAST               3 (err)
            146 DELETE_FAST              3 (err)
            148 LOAD_CONST               0 (None)
            150 RETURN_VALUE
        >>  152 LOAD_CONST               0 (None)
            154 STORE_FAST               3 (err)
            156 DELETE_FAST              3 (err)
            158 RERAISE                  1

 21     >>  160 RERAISE                  0
        >>  162 COPY                     3
            164 POP_EXCEPT
            166 RERAISE                  1
ExceptionTable:
  4 to 88 -> 94 [0]
  94 to 112 -> 162 [1] lasti
  114 to 138 -> 152 [1] lasti
  152 to 160 -> 162 [1] lasti
"""


class Instruction:
    # TODO: __slots__?
    opcode: int
    opname: str
    oparg: int | None
    baseopname: str
    baseopcode: int
    # Offsets in bytes, not code words!
    start_offset: int
    cache_offset: int
    end_offset: int
    jump_target: int | None
    is_jump_target: bool

    def __init__(self, opcode: int, oparg: int, start_offset: int, cache_offset: int):
        self.opcode = opcode
        self.opname = dis.opname[opcode]
        self.oparg = oparg if opcode >= dis.HAVE_ARGUMENT else None
        self.baseopname = dis.deoptmap.get(self.opname, self.opname)
        self.baseopcode = dis.opmap[self.baseopname]
        self.start_offset = start_offset
        self.cache_offset = cache_offset
        self.end_offset = cache_offset + 2 * dis._inline_cache_entries[opcode]
        self.jump_target = None
        if self.baseopcode in dis.hasjrel:
            if self.baseopname == "JUMP_BACKWARD":
                oparg = -oparg
            self.jump_target = self.end_offset + 2 * oparg
        self.is_jump_target = False  # Filled in by parse_bytecode()

    def __repr__(self):
        return f"Instruction{self.__dict__}"


def parse_bytecode(co: bytes) -> list[Instruction]:
    result: list[Instruction] = []
    i = 0
    n = len(co)
    assert n % 2 == 0
    while i < n:
        start_offset = i
        opcode = co[i]
        oparg = co[i + 1]
        i += 2
        while opcode == dis.EXTENDED_ARG:
            opcode = co[i]
            oparg = oparg << 8 | co[i + 1]
            i += 2
        cache_offset = i
        instr = Instruction(opcode, oparg, start_offset, cache_offset)
        i = instr.end_offset
        result.append(instr)
    for instr in result:
        if instr.jump_target is not None:
            for target in result:
                if target.start_offset == instr.jump_target:
                    target.is_jump_target = True
                    break
            else:
                assert False, f"Invalid jump target {instr.jump_target}"

    return result


Stack = tuple[str, ...]


def update_stack(input: Stack, b: Instruction) -> tuple[Stack | None, Stack | None]:
    """Return a pair of optional Stacks.

    If the instruction never jumps, return (Stack, None).
    If it always jumps, return (None, Stack).
    If it may or may not jump, return (StackIfNotJumping, StackIfJumping).
    If it is a return or raise, return (None, None).
    """
    if b.opname in ("RETURN_VALUE", "RETURN_CONST", "RERAISE", "RAISE_VARARGS"):
        return (None, None)
    succ = successors(b)
    if succ == None:
        jumps = [0, None]
    else:
        fallthrough, _ = succ
        if fallthrough:
            jumps = [0, 1]
        else:
            jumps = [None, 1]
    stacks: list[Stack | None] = []
    for jump in jumps:
        if jump is None:
            stacks.append(None)
            continue
        stack = list(input)
        metadata: dict | None = opcode_metadata[b.baseopname]
        popped: int = metadata["popped"](b.oparg, jump)
        pushed: int = metadata["pushed"](b.oparg, jump)
        # print(stack, b.opname, popped, pushed)
        assert popped >= 0 and pushed >= 0, (popped, pushed)
        if len(stack) < popped:
            breakpoint()
            assert False, "stack underflow"
        stack = stack[: len(stack) - popped]
        stack = stack + ["object"] * pushed
        stacks.append(tuple(stack))
    assert len(stacks) == 2
    return tuple(stacks)


def successors(b: Instruction) -> None | tuple[bool, int]:
    assert not dis.hasjabs
    if b.baseopcode not in dis.hasjrel:
        return None
    fallthrough = True
    arg = b.oparg
    assert arg is not None
    if b.baseopname == "JUMP_BACKWARD":
        arg = -arg
        fallthrough = False
    return fallthrough, b.end_offset + 2 * arg


def run(code: types.CodeType):
    # TODO: Break into pieces (maybe make it a class?)
    instrs: list[Instruction] = parse_bytecode(code.co_code)
    assert instrs != []
    stacks: dict[Instruction, Stack | None] = {b: None for b in instrs}
    stacks[instrs[0]] = ()
    todo = True
    etab = dis._parse_exception_table(code)
    for start, end, target, depth, lasti in etab:
        # print(f"ETAB: [{start:3d} {end:3d}) -> {target:3d} {depth} {'lasti' if lasti else ''}")
        for b in instrs:
            if b.start_offset == target:
                b.is_jump_target = True
                stack = ["object"] * depth
                if lasti:
                    stack.append("Lasti")
                stack.append("Exception")
                stacks[b] = tuple(stack)
                break
        else:
            assert False, f"ETAB target {target} not found"
    while todo:
        todo = False
        stack = None
        for b in instrs:
            # print(b.start_offset, b.opname, b.oparg, stack)
            if stack is not None:
                if stacks[b] is None:
                    stacks[b] = stack
                    todo = True
                else:
                    if stacks[b] != stack:
                        print(f"MISMATCH at {b.start_offset}: {stacks[b]} != {stack}")
                        breakpoint()
            else:
                stack = stacks[b]
                if stack is None:
                    continue
            stack_if_no_jump, stack_if_jump = update_stack(stack, b)
            jumps = successors(b)
            if jumps is not None:
                fallthrough, offset = jumps
                # print(b.opname, "JUMP DATA:", fallthrough, offset)
                stack = stack_if_jump
                for bb in instrs:
                    if bb.start_offset == offset:
                        # print("  TO:", bb.opname, "AT", bb.offset)
                        if stacks[bb] is None:
                            stacks[bb] = stack
                            todo = True
                        else:
                            if stacks[bb] != stack:
                                print(f"MISMATCH AT {offset}: {stacks[bb]} != {stack}")
                                breakpoint()
                        break
                else:
                    print(f"CANNOT FIND {offset} for {stack}")
                    breakpoint()
            stack = stack_if_no_jump

    max_stack_size = 4  # len(str(None))
    for stack in stacks.values():
        if stack:
            max_stack_size = max(max_stack_size, len(str(stack)))
    limit = min(max_stack_size, os.get_terminal_size().columns - 40)
    for b in instrs:
        stack = stacks.get(b)
        if stack is not None:
            stack = list(stack)
        prefix = ">>" if b.is_jump_target else "  "
        soparg = f" {b.oparg:{20 - len(b.opname)}d}" if b.oparg is not None else ""
        sstack = str(stack)
        if len(sstack) <= limit:
            print(
                f"{prefix} {sstack:<{limit}s}",
                f"{prefix} {b.start_offset:3d} {b.opname} {soparg}",
            )
        else:
            print(f"{prefix} {sstack}")
            pad = " " * (len(prefix) + limit)
            print(f" {pad} {prefix} {b.start_offset:3d} {b.opname} {soparg}")
        succ = successors(b)
        if (
            b.opname.startswith("RETURN_")
            or "RAISE" in b.opname
            or (succ and not succ[0])
        ):
            print("-" * 40)


if __name__ == "__main__":
    code = test.__code__
    if len(sys.argv[1:]) >= 2:
        module = importlib.import_module(sys.argv[1])
        func = getattr(module, sys.argv[2])
        code = func.__code__
    dis.dis(code)
    run(code)
