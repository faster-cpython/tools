#!/usr/bin/env python3.12

"""Symbolic execution, hero style. :-)"""

import dis
import os
import sys
import types
from typing import Any

import forallcode

import opcode_metadata as opcode_metadata_module

opcode_metadata_module.FVS_MASK = 0x4
opcode_metadata_module.FVS_HAVE_SPEC = 0x4

opcode_metadata: dict[str, dict[str, Any]]
from opcode_metadata import opcode_metadata  # Generated by cases_generator.py

opcode_metadata["COMPARE_AND_BRANCH"]["pushed"] = lambda oparg, jump: 1
opcode_metadata["RETURN_GENERATOR"]["pushed"] = lambda oparg, jump: 1


CO_COROUTINE = 0x00000080
CO_GENERATOR = 0x00000200
CO_ASYNC_GENERATOR = 0x00000800


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
        assert not dis.hasjabs
        if self.baseopcode in dis.hasjrel:
            if self.baseopname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
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


def update_stack(input: Stack, b: Instruction) -> list[tuple[int, Stack]]:
    """Return a list of (int, Stack) pairs corresponding to successors(b).

    If the instruction never jumps, return [(b.end_offset, new_stack)].
    If it always jumps, return [(b.jump_target, new_stack)].
    If it may or may not jump, return [(b.end_offset, new_stack1), (b.jump_target, new_stack2)].
    If it always exits, return [].
    """
    if b.opname in ("RETURN_VALUE", "RETURN_CONST", "RERAISE", "RAISE_VARARGS"):
        return []
    result: list[tuple[int, Stack]] = []
    for offset in successors(b):
        jump = offset is not None
        if not jump:
            offset = b.end_offset
        stack = list(input)
        metadata: dict | None = opcode_metadata[b.baseopname]
        popped: int = metadata["popped"](b.oparg, jump)
        pushed: int = metadata["pushed"](b.oparg, jump)
        # print(" ", stack, b.start_offset, b.opname, b.oparg, "pop=", popped, "push=", pushed)
        assert popped >= 0 and pushed >= 0, (popped, pushed)
        if len(stack) < popped:
            breakpoint()
            assert False, "stack underflow"
        stack = stack[: len(stack) - popped]
        stack = stack + ["object"] * pushed
        result.append((offset, tuple(stack)))
    return result


def successors(b: Instruction) -> list[int | None]:
    """Return a list of successor offsets.

    An offset is either None (fall through to the next instruction) or
    an int (jump to the instruction at that offset, in absolute bytes).

    A "normal" instruction, e.g. LOAD_FAST, returns [None].
    An unconditional jump, e.g. JUMP_BACKWARD, returns [offset].
    A conditional jump, e.g. POP_JUMP_IF_TRUE, returns [None, offset].
    An instruction that only exits, e.g. RETURN_VALUE, returns [].
    """
    assert not dis.hasjabs
    if b.baseopcode not in dis.hasjrel:
        # TODO: Other instructions that have a jump option
        return [None]
    arg = b.oparg
    assert arg is not None
    if b.baseopname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP_FORWARD"):
        # Unconditional jump
        return [b.end_offset - 2 * arg]
    # Conditional jump
    return [None, b.end_offset + 2 * arg]


def run(code: types.CodeType):
    # TODO: Break into pieces (maybe make it a class?)

    # Parse bytecode into Instructions.
    instrs: list[Instruction] = parse_bytecode(code.co_code)
    assert instrs != []

    def instr_from_offset(offset: int) -> Instruction:
        """Map an offset to the Instruction at that offset."""
        # TODO: Use bisect
        for b in instrs:
            if b.start_offset == offset:
                return b
        breakpoint()
        assert False, f"Invalid offset {offset}"

    # Map from Instructions to the stack at the start of the Instruction.
    stacks: dict[Instruction, Stack | None] = {b: None for b in instrs}

    # Initialize stacks with known contents:
    # - The stack at the start of a regular function is empty.
    # - The stack at the start of a generator(-ish) function is ("object",).
    # - The stack at the start of each exception handler is known.
    if code.co_flags & (CO_COROUTINE | CO_GENERATOR | CO_ASYNC_GENERATOR):
        stacks[instrs[0]] = ("object",)
    else:
        stacks[instrs[0]] = ()
    etab = dis._parse_exception_table(code)
    for start, end, target, depth, lasti in etab:
        # print(f"ETAB: [{start:3d} {end:3d}) -> {target:3d} {depth} {'lasti' if lasti else ''}")
        b = instr_from_offset(target)
        b.is_jump_target = True
        stack = ["object"] * depth
        if lasti:
            stack.append("Lasti")
        stack.append("Exception")
        stacks[b] = tuple(stack)

    # Repeatedly propagate stack contents until a fixed point is reached.
    todo = True
    while todo:
        # print("=" * 50)
        todo = False
        for b in instrs:
            stack = stacks[b]
            # print(b.start_offset, b.opname, b.oparg, stack)
            if stack is None:
                continue
            updates = update_stack(stack, b)
            for offset, new_stack in updates:
                bb = instr_from_offset(offset)
                # print("    TO:", bb.opname, "AT", bb.start_offset, "STACK", new_stack)
                if stacks[bb] is None:
                    stacks[bb] = new_stack
                    todo = True
                else:
                    if stacks[bb] != new_stack:
                        print(f"MISMATCH AT {offset}: {stacks[bb]} != {new_stack}")
                        breakpoint()
                        assert False, "mismatch"

    # Format what we found nicely.
    max_stack_size = 4  # len(str(None))
    for stack in stacks.values():
        if stack:
            max_stack_size = max(max_stack_size, len(str(stack)))

    try:
        limit = min(max_stack_size, os.get_terminal_size().columns - 40)
    except OSError:
        limit = 80

    for b in instrs:
        stack = stacks.get(b)
        if stack is not None:
            stack = list(stack)
        prefix = ">>" if b.is_jump_target else "  "
        soparg = (
            f" {b.oparg:{max(1, 20 - len(b.opname))}d}" if b.oparg is not None else ""
        )
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
        if (
            b.opname.startswith("RETURN_")
            or "RAISE" in b.opname
            or (None not in successors(b))
        ):
            print("-" * 40)


def main():
    if sys.argv[1:]:
        for code in forallcode.find_all_code(sys.argv[1:], 1):
            print()
            print(code)
            dis.dis(code, adaptive=True, depth=0, show_caches=False)
            run(code)
    else:
        dis(test)
        run(test.__code__)


if __name__ == "__main__":
    main()
