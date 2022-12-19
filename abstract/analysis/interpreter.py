"""
This is the core of the analysis.

You have the interpreters for the different abstract analysis that act
on two different levels

 1. the blocks level: if/while blocks
 2. the instructions level
"""
from abc import ABC, abstractmethod
from typing import Type, TypeVar, List, Dict

from .code import Function, WhileBlock, IfBlock, CodeBlock, BaseBlock, ContiguousBlock, Label, BreakBlock
from ..utils import LoggerMixin

B = TypeVar('B', bound=BaseBlock)


class BaseInterpreter(ABC):
    """Provide the basic methods to handle different instructions."""

    def interpret(self, head: CodeBlock, *args, **kwargs):
        self.do_blocks(head, *args, **kwargs)

    # This is intended to be overridden and have a "match" inside
    # to use pattern matching in order to be faster
    @abstractmethod
    def do_instructions(self, block: B, op: Label):
        """Manage the single instructions."""

    def do_blocks(self, block: B, *args, **kwargs):
        match block:
            case IfBlock():
                self.do_ifblock(block, *args, **kwargs)
            case WhileBlock():
                self.do_whileblock(block, *args, **kwargs)
            case ContiguousBlock():
                self.do_contiguousblock(block, *args, **kwargs)
            case CodeBlock():
                self.do_codeblock(block, *args, **kwargs)
            case BreakBlock():
                self.do_breakblock(block, *args, **kwargs)
            case _:
                raise ValueError("unexpected block type: %s", block)

    def do_codeblock(self, block: CodeBlock, *args, **kwargs):
        for l in block.ops:
            self.do_instructions(block, l)

    @abstractmethod
    def do_breakblock(self, block: BreakBlock, *args, **kwargs):
        pass

    def do_contiguousblock(self, block: ContiguousBlock, *args, **kwargs):
        self.do_blocks(block.sl, *args, **kwargs)
        self.do_blocks(block.s, *args, **kwargs)

    def do_ifblock(self, ifblock: IfBlock, *args, **kwargs):
        self.do_ifheader(ifblock._head, True, *args, **kwargs)
        self.do_ifbody(ifblock._body[0], True, *args, **kwargs)

        if ifblock._body[1]:
            # Note the branch parameter
            self.do_ifheader(ifblock._head, False, *args, **kwargs)
            self.do_ifbody(ifblock._body[1], False, *args, **kwargs)

    def do_ifbody(self, body, branch: bool, *args, **kwargs):
        self.interpret(body, *args, **kwargs)

    @abstractmethod
    def do_ifheader(self, header: CodeBlock, branch: bool, *args, **kwargs):
        pass

    def do_whileblock(self, whileblock, *args, **kwargs):
        self.do_whileheader(whileblock._head, *args, **kwargs)
        self.do_whilebody(whileblock._body)

    def do_whilebody(self, body: CodeBlock, *args, **kwargs):
        self.interpret(body, *args, **kwargs)

    @abstractmethod
    def do_whileheader(self, whileblock, *args, **kwargs):
        pass

    @abstractmethod
    def result(self):
        pass


class CodeInterpreter(BaseInterpreter, LoggerMixin):

    def result(self):
        return self.get_lines()

    def __init__(self):
        self.lines = []
        self.level = 0

    def get_lines(self) -> List[str]:
        return self.lines

    def indent(self, count, lines: List[str]):
        return list(map(
            lambda _: "{}{}".format(" " * count, _),
            lines,
        ))

    def increase_indent(self):
        self.level += 1

    def decrease_indent(self):
        self.level -= 1

    def add_lines(self, lines: List[str]):
        self.lines.extend(self.indent(self.level * 4, lines))

    def do_instructions(self, block: B, op: Label):
        self.add_lines([op])

    def do_breakblock(self, block: BreakBlock, *args, **kwargs):
        self.add_lines(['break'])

    def do_whilebody(self, body: CodeBlock, *args, **kwargs):
        self.increase_indent()

        super().do_whilebody(body, *args, **kwargs)

        self.decrease_indent()
        self.add_lines(["}"])

    def do_ifbody(self, body: CodeBlock, branch, *args, **kwargs):
        self.increase_indent()

        super().do_ifbody(body, branch, *args, **kwargs)

        self.decrease_indent()

        if not branch:
            self.add_lines(["}"])

    def do_ifheader(self, header: CodeBlock, branch: bool, *args, **kwargs):
        self.add_lines(["if ({:s}) {{".format(str(header.ops[-1])) if branch else "} else {"])

    def do_whileheader(self, whilehead, *args, **kwargs):
        self.logger.debug(whilehead.ops)
        self.add_lines(["while ({}) {{".format(str(whilehead.ops[-1]))])

    def interpret(self, head: CodeBlock, *args, **kwargs):
        super().interpret(head, *args, **kwargs)
