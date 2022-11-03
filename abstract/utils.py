import dataclasses
import logging
from collections import namedtuple, deque, defaultdict
from typing import Deque, Any, List, Iterable, TypeVar, Tuple

from . import currentProgram, getInstructionAt
from .pcode import Op, Variable

from ghidra.app.decompiler import DecompileOptions
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor


import pygraphviz as pgv


logger = logging.getLogger(__name__)


def decompile(func, program=None):
    monitor = ConsoleTaskMonitor()
    ifc = DecompInterface()
    options = DecompileOptions()
    ifc.setOptions(options)

    program = program or currentProgram

    ifc.openProgram(program)

    res = ifc.decompileFunction(func, 60, monitor)

    return res


class BasicBlock:
    """Wrap the ghidra's basic block representation."""
    def __init__(self, pcodeblock):
        self.start = pcodeblock.getStart()
        self.end = pcodeblock.getStop()

        self._block = pcodeblock

    def __str__(self):
        return '@{}\n\n{}'.format(self.start, "\n".join([str(_) for _ in self.pcodes()]))

    # __members, __eq__ and __hash__ are necessary in order
    # to make this class hashable and be able to compare elements
    # for example in a dictionary
    def __members(self) -> Tuple[Any, Any]:
        return self.start, self.end

    def __eq__(self, other):
        if type(self) == type(other):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())

    @property
    def ins(self):
        return [self.__class__(self._block.getIn(_)) for _ in range(self._block.getInSize())]

    @property
    def outs(self) -> List['BasicBlock']:
        return [self.__class__(self._block.getOut(_)) for _ in range(self._block.getOutSize())]

    @property
    def true(self):
        return self.outs[1] if len(self.outs) > 1 else None

    @property
    def false(self):
        return self.outs[0] if len(self.outs) > 0 else None

    def instructions(self):
        inst = getInstructionAt(self.start)

        while inst and inst.getAddress() <= self.end:
            yield inst

            inst = inst.getNext()

    def _raw_pcodes(self):
        for inst in self.instructions():
            for op in inst.getPcode():
                yield Op(op)

    def _pcodes(self):
        """Return the elaborated Pcodes from the basic blocks.

        If you want the raw pcodes (i.e. without the pre-analysis from
        ghidra itself) you should call raw_pcodes()."""
        it = self._block.getIterator()

        for op in it:
            yield Op(op)

    def pcodes(self, raw: bool = False):
        if raw:
            return self._raw_pcodes()
        else:
            return self._pcodes()


T = TypeVar("T")


def traverse(graph: T) -> Iterable[T]:
    """Traverse a "generic" directed graph"""

    visited = {}

    queue: Deque[T] = deque()

    queue.append(graph)

    while queue:
        block = queue.popleft()
        visited[block] = True

        yield block

        for child in block.outs:
            if child not in visited:  # this is fragile, for a custom class you need to be careful
                queue.append(child)


class CFG:
    """Wrap the BasicBlock CFG in order to do operations."""
    def __init__(self, starting_block: BasicBlock):
        self._blocks = starting_block

    def traverse(self) -> Iterable[BasicBlock]:
        return traverse(self._blocks)

    def build(self, raw=False):
        variables = {}

        def _get_name():
            count = 0

            while True:
                yield "var%d" % count

                count += 1

        names = _get_name()

        for block in self.traverse():
            for op in block.pcodes(raw=raw):
                print("op:", op)
                inputs = op.inputs
                output = op.output

                logger.debug("inputs: %s", inputs)
                logger.debug("output: %s", output)


class Function:
    """Wrap the representation of a function.

    In ghidra there are two main representations, Function and HighFunction."""
    def __init__(self, high_function):
        self._high = high_function
        self.cfg = CFG(BasicBlock(self._high.getBasicBlocks()[0]))  # TODO

    def graphivz(self) -> None:
        logger.info("graphivz for ")

        cfg = pgv.AGraph(directed=True)
        cfg.node_attr.update(shape="note")

        for block in self.cfg.traverse():
            logger.debug("block@%s", block.start)
            for out in block.outs:
                cfg.add_edge(block, out)

        logger.debug(cfg)

        cfg.layout('dot')
        cfg.draw("cfg.png")
