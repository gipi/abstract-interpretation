"""Module to handle graph related routines."""
from __future__ import annotations

from collections import defaultdict
from enum import Flag, auto
from functools import cached_property
from typing import Tuple, Dict, MutableMapping, Any, List, Protocol, Set

from ..pcode import Op
from ..utils import traverse, NodeType, LoggerMixin


class CFGNode(Protocol):
    start: int
    n_ins: int
    n_outs: int
    ins: List[CFGNode]
    outs: List[CFGNode]


class BasicBlock(LoggerMixin):
    """Wrap the ghidra's basic block representation."""
    def __init__(self, pcodeblock: PcodeBlockBasic):
        self.start: int = pcodeblock.getStart().getOffset()
        self.end = pcodeblock.getStop().getOffset()

        self._block = pcodeblock
        self.n_ins = self._block.getInSize()
        self.n_outs = self._block.getOutSize()

        self._code = None  # this will contain the eventual High level code block

    def __str__(self):
        # return '@{}\n\n{}'.format(self.start, "\n".join([str(_) for _ in self.pcodes()]))

        return '@{:x}'.format(self.start)

    def __repr__(self):
        return "<{}(@{:x})>".format(self.__class__.__name__, self.start)

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

    def set_code(self, block):
        # TODO: is it a problem for GC (since block references this object)?
        self._code = block

    @cached_property
    def ins(self) -> List['BasicBlock']:
        return [self.__class__(self._block.getIn(_)) for _ in range(self.n_ins)]

    @cached_property
    def outs(self) -> List['BasicBlock']:
        return [self.__class__(self._block.getOut(_)) for _ in range(self.n_outs)]

    @property
    def true(self):
        return self.outs[1] if len(self.outs) > 1 else None

    @property
    def false(self):
        return self.outs[0] if len(self.outs) > 0 else None

    def instructions(self):  # getFlow() can tell if jump
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


class EdgeType(Flag):
    TREE = auto()
    FORWARD = auto()
    CROSS = auto()
    BACK = auto()


EdgeClassification = MutableMapping[Tuple[CFGNode, CFGNode], EdgeType]
Forest = List[CFGNode]


def DFS(v: CFGNode, edges: EdgeClassification, index: int, num: Dict[CFGNode, int], mark: Dict[CFGNode, int], loops: Set[CFGNode]):
    """Depth first search algorithm to find the "depth first spanning forest".

    Take in mind that the classification of the vertices depends on the ordering of the nodes themselves
    (but in our case is not big deal since we are more interested in the back-edges that are invariant)."""
    index += 1  # used to number the vertices

    num[v] = index
    mark[v] = 1

    # here the core of the classification
    for w in v.outs:
        if num[w] == 0:
            edges[(v, w)] = EdgeType.TREE  # discovers the vertex for the first time
            DFS(w, edges, index, num, mark, loops)
        elif num[w] > num[v]:
            edges[(v, w)] = EdgeType.FORWARD  # it's not the first time
        elif mark[w] == 0:
            edges[(v, w)] = EdgeType.CROSS  # vertex w it's in another tree
        else:
            edges[(v, w)] = EdgeType.BACK  # connects back to an ancestor (it's a loop)
            loops.add(w)

    mark[v] = 0


def classify(head: CFGNode) -> Tuple[Forest, EdgeClassification, Set[CFGNode]]:
    """Returns the classification of the directed edges and the root nodes
    of the forest tree of the CFG represented by the parameter "head"
    (the entry point of the function)."""
    edges: EdgeClassification = dict()
    num: Dict[CFGNode, int] = defaultdict(int)
    mark: Dict[CFGNode, int] = defaultdict(int)
    forest = []
    loops = set()

    index = 0  # initialize a variable for the index

    for kind, node in traverse(head):
        if kind != NodeType.NEW:
            continue

        if num[node] == 0:
            forest.append(node)
            DFS(node, edges, index, num, mark, loops)

    return forest, edges, loops
