"""Module to handle graph related routines."""
from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict, deque
from enum import Flag, auto
from functools import cached_property
from typing import Tuple, Dict, MutableMapping, Any, List, Protocol, Set

import pygraphviz as pgv

from ..utils import traverse, NodeType, LoggerMixin


logger = logging.getLogger(__name__)


class CFGNode(Protocol):
    start: int
    n_ins: int
    n_outs: int
    ins: List[CFGNode]
    outs: List[CFGNode]
    gotos: List[CFGNode]


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
                yield op

    def _pcodes(self):
        """Return the elaborated Pcodes from the basic blocks.

        If you want the raw pcodes (i.e. without the pre-analysis from
        ghidra itself) you should call raw_pcodes()."""
        it = self._block.getIterator()

        for op in it:
            self.logger.debug(op)
            yield op

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


def DFS(v: CFGNode, edges: EdgeClassification, index: int, num: Dict[CFGNode, int],
        mark: Dict[CFGNode, int], loops: Set[CFGNode]) -> None:
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
    loops: Set[CFGNode] = set()

    index = 0  # initialize a variable for the index

    for kind, node in traverse(head):
        if kind != NodeType.NEW:
            continue

        if num[node] == 0:
            forest.append(node)
            DFS(node, edges, index, num, mark, loops)

    return forest, edges, loops


CFGNodeToPaths = Dict[CFGNode, List[str]]
CFGPathToNodes = Dict[str, List[CFGNode]]


@dataclasses.dataclass
class CFGLoop:
    """Terse representation of the head of a while loop."""
    head: CFGNode
    body: CFGNode
    exit: CFGNode
    path: str


CFGLoops = List[CFGLoop]


def explore(head: CFGNode, paths: CFGNodeToPaths, nodes: CFGPathToNodes, loops: CFGLoops, prefix: str = ""):
    """Recursively build all the binary paths of the graph.

    The idea here is that, since every node of a CFG has at most two possible paths
    when exiting, we can create a list of paths passing through a node appending
    a "0" or "1" depending on the child choose along the way.

    The immediate benefit is that, when passing through a node that has already a
    path that prefixes perfectly the one we are on, then we have a loop and the fist
    digit after the piece that matches is the branch that points to the body of the loop.

    This allows us to immediately know that the other branch is the exiting condition of
    the loop, where a "break" statement would point. To find possible nodes with an edge
    directed to this node we can simply get the paths having as prefix the path inside the
    body of the loop that pass through the exit node.
    """
    # first of all save the path
    paths_node = paths[head]
    old_paths_node = paths_node[:]
    paths_node.append(prefix)

    # and the node
    nodes[prefix].append(head)

    # if a previous path is a prefix of this one we have a loop
    for previous_path in old_paths_node:
        if not prefix.startswith(previous_path):
            continue

        if head.start not in [_.head.start for _ in loops]:
            logger.info("found loop @%s ('%s' prefixes '%s')", head, previous_path, prefix)

            overlap = len(previous_path)
            # we use the first different digit after the prefix
            # to know which child is the body
            direction = int(prefix[overlap:overlap + 1])
            # we save the shortest path that enters the body
            # so to recognize immediately what path reaches
            # a "break" statement
            path_body = prefix[: overlap + 1]
            loops.append(
                CFGLoop(
                    head,
                    head.outs[direction],
                    head.outs[~direction & 1],
                    path_body,
                ),
            )

        return

    if head.n_outs == 0:  # no child no party
        return

    if head.n_outs == 1:  # 1 child the prefix remain the same
        return explore(head.outs[0], paths, nodes, loops, prefix=prefix)

    explore(head.outs[0], paths, nodes, loops, prefix=prefix + "0")
    explore(head.outs[1], paths, nodes, loops, prefix=prefix + "1")


def explore_all(head: CFGNode) -> Tuple[Dict[CFGNode, List[str]], Dict[str, List[CFGNode]], List[CFGLoop]]:
    paths: Dict[CFGNode, List[str]] = defaultdict(list[str])
    nodes: Dict[str, List[CFGNode]] = defaultdict(list[CFGNode])
    loops: List[CFGLoop] = list()

    explore(head, paths, nodes, loops)

    return paths, nodes, loops


def graphivz(name_func, head: CFGNode) -> None:
    logger.info("graphivz for '%s'", name_func)

    cfg = pgv.AGraph(directed=True)
    cfg.node_attr.update(shape="note")

    forest, edges, _ = classify(head)

    for v, w in edges.keys():
        cfg.add_edge(v, w)

    logger.debug(cfg)

    cfg.layout('dot')

    name_output = "cfg-{}.png".format(name_func)
    logger.info("dumped image: '%s'", name_output)
    cfg.draw(name_output)
