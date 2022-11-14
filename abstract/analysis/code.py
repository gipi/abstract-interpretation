"""
Main module for the analysis as described by "Principles of abstract interpretation"
"""
from __future__ import annotations

import itertools
import logging
import operator
from abc import ABC, abstractmethod
from itertools import permutations
from typing import List, Dict, Optional, Tuple, Generator, Union, Protocol, Set, TypedDict

import pygraphviz as pgv

from ghidra.app.decompiler import DecompileOptions
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

from .graph import BasicBlock, CFGNode, classify
from ..utils import traverse, NodeType, LoggerMixin
from .. import currentProgram
from ..pcode import Op


def decompile(func, program=None):
    monitor = ConsoleTaskMonitor()
    ifc = DecompInterface()
    options = DecompileOptions()
    ifc.setOptions(options)
    program = program or currentProgram
    ifc.openProgram(program)
    res = ifc.decompileFunction(func, 60, monitor)
    return res


class PlainNode(TypedDict):
    ins: List[int]
    outs: List[int]
    code: List[str]


PlainCFG = Dict[int, PlainNode]


class CFG(LoggerMixin):
    """Wrap the ghidra's BasicBlock CFG in order to do operations."""
    def __init__(self, starting_block: BasicBlock):
        self._blocks = starting_block

    def traverse(self, entry: BasicBlock = None) -> Generator[BasicBlock, None, None]:
        entry = self._blocks if entry is None else entry

        for kind, block in traverse(entry):
            if kind != NodeType.NEW:
                continue

            yield block

    def ops(self, raw=False):

        for block in self.traverse():
            print("block @{:x}".format(block.start.getOffset()))
            for op in block.pcodes(raw=raw):
                print("op:", op)
                inputs = op.inputs
                output = op.output

                self.logger.debug("inputs: %s", inputs)
                self.logger.debug("output: %s", output)

    def linearize(self):
        """Restructure the CFG in while and if using a "recursive" approach."""
        # we start by traversing the tree
        block = self._blocks

        blocks = []  # stack for blocks found

        while block:
            outs = block.outs
            n_outs = block.n_outs
            if n_outs != 2:
                self.logger.info("code block @%x", block.start.getOffset())
                # it's a "linear block" we can save it and goint to the next
                blocks.append(block)
                block = None if n_outs == 0 else outs[0]
                continue

            # there are two possibilities
            # it's an if or a while
            hblock, block = self.ifs(entry=block)
            blocks.append(hblock)

            self.logger.info("new code block %s", hblock)

        breakpoint()

        return blocks

    def ifs(self, entry: BasicBlock = None) -> Tuple[Optional[IfBlock, WhileBlock], BasicBlock]:
        """Recognize If/while blocks.

        This method returns the "high level" block if it exists together with
        the next block from which continue "parsing"."""
        # we start from an entry point that we split immediately
        # and travel them in parallel in order to
        # 1. find a common node
        #  '-> it's an "if" and the exit node it's this common node
        # 2. find again the entry point
        #  '-> it's a "while" and the first node after the head in the other branch is the exit node
        # NOTE: might happen that "if" inside a "while" has the confluent node
        #       much far away from one side wrt the other so to have the "parent"
        #       node recognize as the head of the loop. To avoid this we can take
        #       into account that when we enter a bifurcation we need to terminate
        #       at the node where two paths converge
        self.logger.debug("looking for a code block starting @%x", entry.start.getOffset())

        # now check that one flows into the other
        left_head = entry.outs[0]
        right_head = entry.outs[1]

        self.logger.debug("left head: @%x", left_head.start.getOffset())
        self.logger.debug("right head: @%x", right_head.start.getOffset())

        left = []
        right = []

        l_block, r_block = left_head, right_head

        # now we procede in parallel
        while l_block or r_block:
            if r_block:
                self.logger.debug("right: @%x", r_block.start.getOffset())

                r_result = self.analyze_one_side(r_block, right, left, entry)

                if isinstance(r_result, tuple):
                    return r_result

                r_block = r_result

            if l_block:
                self.logger.debug("left: @%x", l_block.start.getOffset())

                l_result = self.analyze_one_side(l_block, left, right, entry)

                if isinstance(l_result, tuple):
                    return l_result

                l_block = l_result
        else:
            raise Exception("No exit node for the IF found")

    def json(self) -> PlainCFG:
        """Returns a JSON representation of the Pcode basic blocks."""
        output = {}
        for block in self.traverse():
            ins = list(map(operator.attrgetter('start'), block.ins))
            outs = list(map(operator.attrgetter('start'), block.outs))
            code = list(map(str, block.pcodes()))

            node = {
                'ins': ins,
                'outs': outs,
                'code': code,
            }

            output[block.start] = node

        return output


class Function(LoggerMixin):
    """Wrap the representation of a function.

    In ghidra there are two main representations, Function and HighFunction."""
    def __init__(self, high_function):
        self._high = high_function
        entry = BasicBlock(self._high.getBasicBlocks()[0])

        # TODO: assert len(forest) == 1
        self.forest, self.edges, self.loops = classify(entry)

        self.logger.debug(self.edges)

        StructureCode(entry, self.loops)
        self.cfg = CFG(entry)  # TODO

    def graphivz(self) -> None:
        name_func = self._high.getFunction().getName()
        self.logger.info("graphivz for '%s'", name_func)

        cfg = pgv.AGraph(directed=True)
        cfg.node_attr.update(shape="note")

        for block in self.cfg.traverse():
            self.logger.debug("block@%s", block.start)
            for out in block.outs:
                cfg.add_edge(block, out)

        self.logger.debug(cfg)

        cfg.layout('dot')

        name_output = "cfg-{}.png".format(name_func)
        self.logger.info("dumped image: '%s'", name_output)
        cfg.draw(name_output)

    @classmethod
    def from_name(cls, name: str, program=None):
        high = decompile(name, program or currentProgram).getHighFunction()

        return cls(high)


class CodeComponent(Protocol):
    @classmethod
    def detect(cls, node: CFGNode) -> Optional[CFGNode]:
        pass


logger_components = logging.getLogger("{}.{}".format(__name__, "components"))


class BaseBlock(ABC):

    @abstractmethod
    def structure_ins_for_outs(self, outs: List[CFGNode]):
        """The class derived from this are used to create a linear
        flow of code, so the entering edge of the out node must be
        adjusted to reflect the transformation after this node creation."""
        pass

    @abstractmethod
    def structure_outs_for_ins(self, ins: List[CFGNode]):
        """The class derived must fix the edge entering in it"""
        pass


class SimpleBaseBlockInsMixin:
    def helper_structure_outs_for_ins(self, head: CFGNode, ins: List[CFGNode]):
        for entering in ins:
            outs = entering.outs[:]
            for index, out in enumerate(outs):
                if out == head:
                    entering.outs[index] = self
                    break


class SimpleBaseBlockOutsMixin:
    def helper_structure_ins_for_outs(self, head: CFGNode, outs: List[CFGNode]):
        for exiting in outs:
            ins = exiting.ins[:]
            for index, entering in enumerate(ins):
                if entering == head:
                    exiting.ins[index] = self
                    break


class WhileBlock(BaseBlock, SimpleBaseBlockInsMixin, SimpleBaseBlockOutsMixin):
    """Represents a set of blocks implementing a loop.

    The diagram for this kind of block is the following

      ,<head>
     |   /  \
    |   /  <first block out of the loop>
    -<body>
    """

    def structure_ins_for_outs(self, outs: List[CFGNode]):
        self.helper_structure_ins_for_outs(self._head, outs)

    def structure_outs_for_ins(self, ins: List[CFGNode]):
        self.helper_structure_outs_for_ins(self._head, ins)

    def __init__(self, head: CFGNode, body: CFGNode, out: CFGNode):
        self._head = head
        self._body = body
        self.start = self._head.start

        # now we want to reconstruct the ins/outs
        # the ins are the ins of the head but with the edge from the body removed
        self.ins = list(set(self._head.ins) - {self._body})
        self.structure_outs_for_ins(self.ins)
        self.n_ins = len(self.ins)

        self.outs = [out]
        self.structure_ins_for_outs(self.outs)
        self.n_outs = 1

    def __repr__(self):
        return "<{}(head=@{:x}, exit=@{:x})>".format(
            self.__class__.__name__,
            self._head.start,
            self.outs[0].start,
        )

    def __contains__(self, item):
        return item is self._head or item in self._body

    def is_head(self, block):
        return block is self._head

    @property
    def head(self):
        return self._head

    def code(self, indent=0):
        head = ["while (<something>) {"]
        body = self._body.code(indent=4)
        tail = ["}"]

        return list(map(
            lambda _: "{}{}".format(" " * indent, _),
            itertools.chain.from_iterable([head, body, tail]),
        ))

    @classmethod
    def detect(cls, node: CFGNode) -> Optional[WhileBlock]:
        if node.n_outs != 2:
            return None
        # loop over the combinations of the exit nodes
        # a is the body of the loop and b is the exit node
        for a, b in permutations(node.outs, 2):
            if a.n_outs != 1:  # the body has one exit node
                continue

            if a.outs[0] != node:  # the body come back to the head
                continue

            logger_components.debug("detected WhileBlock at @%x", node.start)
            return cls(node, a, b)

        return None


class IfBlock(BaseBlock):

    def __init__(self, head: CFGNode, left: CFGNode, right: Optional[CFGNode], out: CFGNode):
        self._head = head
        self._body = [left, right]
        self.start = self._head.start

        self.n_outs = 1
        self.outs = [out]

        self.n_ins = self._head.n_ins
        self.ins = self._head.ins

        # fix the ins and outs
        self.structure_ins_for_outs(self.outs)
        self.structure_outs_for_ins(self.ins)

    def structure_ins_for_outs(self, outs: List[CFGNode]):
        # remove the left and right edges from the exit node
        # and add the IfBlock itself
        out = outs[0]  # we have only one exit node
        out_ins = set(out.ins) - {self._body[0]}
        if self._body[1]:
            out_ins = out_ins - {self._body[1]}
        else:
            out_ins = out_ins - {self._head}

        out_ins = out_ins | {self}

        out.ins = list(out_ins)

    def structure_outs_for_ins(self, ins: List[CFGNode]):
        for entering in ins:
            outs = entering.outs[:]
            for index, out in enumerate(outs):
                if out == self._head:
                    entering.outs[index] = self
                    break

    def __repr__(self):
        return "<{}(@{:x})>".format(self.__class__.__name__, self._head.start)

    def code(self, indent: int = 0) -> List[str]:
        """Display the linearized code"""
        head = ["if (<something>) {"]
        true = self._body[0].code(indent=4)
        false = itertools.chain.from_iterable([["} else {"], self._body[1].code(indent=4)]) if self._body[1] else []
        tail = ["}"]

        return list(map(
            lambda _: "{}{}".format(" " * indent, _),
            itertools.chain.from_iterable([head, true, false, tail]),
        ))

    @classmethod
    def detect(cls, node: CFGNode) -> Optional[IfBlock]:
        # first of all we need a branch
        if node.n_outs != 2:
            return None

        # loop over the combinations of the exit nodes
        for a, b in permutations(node.outs, 2):
            # we consider b the true condition
            if b.n_outs != 1:
                continue
            if b.n_ins != 1:
                continue

            # we consider b -> a, i.e. an "if" without an "else"
            if b.outs[0] == a and a.n_ins >= 2:
                logger_components.debug("found IfBlock for %s", node)
                return cls(node, b, None, a)

            if a.n_outs != 1 or a.n_ins != 1:
                continue

            if a.outs[0] == b.outs[0]:
                logger_components.debug("detected IfBlock (with else) for %s", node)
                return cls(node, b, a, a.outs[0])

        else:
            return None


class ContiguousBlock(BaseBlock, SimpleBaseBlockInsMixin, SimpleBaseBlockOutsMixin):
    """Aggregate together blocks that are contiguous."""

    def structure_ins_for_outs(self, outs: List[CFGNode]):
        self.helper_structure_ins_for_outs(self._blocks.outs[0], outs)

    def structure_outs_for_ins(self, ins: List[CFGNode]):
        self.helper_structure_outs_for_ins(self._blocks, ins)

    def __init__(self, blocks: CFGNode, outs: List[CFGNode]):
        self._blocks = blocks
        self.start = self._blocks.start

        self.outs = outs
        self.ins = self._blocks.ins

        self.structure_outs_for_ins(self.ins)
        self.structure_ins_for_outs(self.outs)

        self.n_outs = len(self.outs)
        self.n_ins = len(self.ins)

    def __repr__(self):
        return "<{}(@{:x})>".format(self.__class__.__name__, self.start)

    def code(self, indent=0):
        return list(map(
            lambda _: "{}{}".format(" " * indent, _),
            self._blocks.code() + self._blocks.outs[0].code(),
        ))

    @classmethod
    def detect(cls, node: CFGNode) -> Optional[ContiguousBlock]:
        # we must have only one exit edge
        if node.n_outs != 1:
            return None

        out = node.outs[0]

        # this edge must have us as only predecessor
        if out.n_ins != 1:
            return None

        # then we can unite them together
        logger_components.debug("detected ContiguousBlock @%x", node.start)
        return cls(node, out.outs)


class CodeBlock(LoggerMixin):
    """Wrap the ghidra's blocks in order to allow for reworking of the CFG
    with higher-level abstraction."""

    def __init__(self, start: int, block: BasicBlock = None, ins: List[CFGNode] = None, outs: List[CFGNode] = None):
        self._block = block
        self.start = start
        self.ins = ins
        self.outs = outs

    def __repr__(self):
        return "<{}(@{:x})>".format(self.__class__.__name__, self.start)

    @property
    def n_ins(self):
        return len(self.ins)

    @property
    def n_outs(self):
        return len(self.outs)

    def code(self, indent: int = 0) -> List[str]:
        return ["{}@{:x}".format(" " * indent, self.start)]


class StructureCode(LoggerMixin):
    """Main component used to restructure the CFG to code blocks."""
    COMPONENTS: List[CodeComponent] = [
        ContiguousBlock,
        IfBlock,
        WhileBlock,
    ]

    def __init__(self, cfg: CFGNode, loops: Set[CFGNode]):
        self.loops = loops

        changed, new_head = self.linearize(cfg, dict(), self.reset_loops_traversing())
        self.logger.debug(new_head)

        while changed:
            changed, new_head = self.linearize(new_head, dict(), self.reset_loops_traversing())
            self.logger.debug("new head: %s, changed: %s", repr(new_head), changed)

        self._head = new_head

    def get(self):
        return self._head

    def reset_loops_traversing(self) -> Dict[int, bool]:
        # create a dictionary with the starting address of the block
        # containing the head of the loop indicating if we have traversed
        # such block already
        loops_traversed = {}
        for head in self.loops:
            loops_traversed[head.start] = False

        return loops_traversed

    def linearize(self, node: CFGNode, history: Dict[int, CFGNode], loops_traversed: Dict[int, bool]) -> Tuple[bool, CFGNode]:
        """Linearizes the given CFG starting from the block "node".

        This method returns a tuple where the first element is the boolean indicating
        if some substitution happened and the other element is the root node of the linearized
        CFG (that might be the same exact node passed as input)."""
        # the idea is to recursively try to find if components match
        # the given node or try again with its children
        logger_components.debug("linearize: %s", repr(node))
        changed = False
        for index in range(node.n_outs):
            child = node.outs[index]

            if child.start in loops_traversed:
                traversed = loops_traversed[child.start]

                if traversed:
                    continue

                loops_traversed[child.start] = True

            # then we try to linearize it
            this_changed, found = self.linearize(child, history, loops_traversed)

            if this_changed:
                changed = True

        for component in self.COMPONENTS:
            found = component.detect(node)

            if found:
                return True, found

        return changed, node

    def identify_loops(self) -> List[List[BasicBlock]]:
        """Identify the blocks that form a loop.

        The idea is to find all the loops and then isolate one at the times
        starting from the shortest.

        Each candidate is enlarged to include all the nodes that flow into
        the main body."""

        loops: List[List[BasicBlock]] = []
        ancestors = []

        for kind, block in traverse(self._head, ancestors=ancestors):
            self.logger.debug("block: @%s type: %s", hex(block.start.getOffset()), kind)
            self.logger.debug("ancestors: %s", [hex(_.start.getOffset()) for _ in ancestors])

            if kind == NodeType.ANCESTOR:
                self.logger.debug("found loop: %s", ancestors)
                # index of the first occurrence of the block (it's the head of the loop)
                ancestor_index = ancestors.index(block)
                # we save it (copying it)
                loops.append(ancestors[ancestor_index:])

        tmp = []

        # sort loops by length
        for loop in loops:
            tmp.append((len(loop), loop))
            self.logger.debug("loop with length %d: %s" % (
                len(loop),
                [hex(_.start.getOffset()) for _ in loop],
            ))

        # TODO: note that probably it's not enough to sort in this way

        return list(map(
            operator.itemgetter(1),
            sorted(tmp, key=operator.itemgetter(0)),
        ))

    def analyze_one_side(self, block: BasicBlock, side: List[BasicBlock], other: List[BasicBlock], head: BasicBlock) \
            -> Optional[Tuple[WhileBlock, BasicBlock], BasicBlock, None]:
        if block == head:
            self.logger.debug("loop detected")
            return WhileBlock(head, side), other[0]

        if block in other:
            self.logger.debug("found tail")
            index_tail = other.index(block)
            return IfBlock(head, other[:index_tail], side), block

        if False and block.n_ins == 2:
            # this block is the terminal for a code block
            # save it and wait for the other side
            side.append(block)
            return None

        if block.n_outs == 2:
            hblock, block = self.ifs(block)
            side.append(hblock)
            return block

        side.append(block)
        return block.outs[0] if block.n_outs > 0 else None


class Trace:
    """Represent an execution trace, i.e. a sequence of operations between two labels.

    In this implementation the labels are the sequence numbers of pcodes."""
    def __init__(self):
        self._operations: List[Op] = []

    def action(self, op: Op):
        self._operations.append(op)
