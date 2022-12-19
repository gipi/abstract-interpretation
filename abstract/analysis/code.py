"""
Main module for the analysis as described by "Principles of abstract interpretation"
"""
from __future__ import annotations

import itertools
import logging
import operator
import random
from abc import ABC, abstractmethod
from itertools import permutations
from typing import List, Dict, Optional, Tuple, Generator, Protocol, TypedDict, Set

from .graph import BasicBlock, CFGNode, explore_all, CFGPathToNodes, CFGLoop, CFGNodeToPaths, CFGLoops
from ..utils import traverse, NodeType, LoggerMixin
from ..pcode import Expression, serialize_expression, Variable, deserialize_expressions


class Provider(ABC):
    """Translate a description of binary-level instruction to our AST."""

    @abstractmethod
    def get_start(self) -> int:
        """Return the address of the start block"""

    @abstractmethod
    def cfg(self) -> CodeBlock:
        pass

    @abstractmethod
    def variables(self) -> List[Variable]:
        pass


class PlainNode(TypedDict):
    ins: List[int]
    outs: List[int]
    code: List[Tuple]


PlainCFG = Dict[int, PlainNode]


class PlainAST(TypedDict):
    vars: List[Tuple]
    start: int
    cfg: PlainCFG


class PlainProvider(Provider, LoggerMixin):
    """Translate from a serialized dump"""

    def __init__(self, json: PlainAST):
        self.json = json

    def get_start(self) -> int:
        return self.json['start']

    def get_head(self) -> PlainNode:
        return self.json['cfg'][self.get_start()]

    def cfg(self) -> CodeBlock:
        """Build the CFG of the function"""

        # The procedure is kind of long since we want to create
        # a CFG where each node is a unique instance in order
        # to be able to create (at a later stage) structure code
        # in a more sane way
        address_start = self.get_start()

        nodes = self.json['cfg']

        blocks: Dict[int, CodeBlock] = {}
        # now we can build the high-level CFG
        # first creating the nodes without in/out edges
        for start in nodes:
            node: PlainNode = nodes[start]
            code = [deserialize_expressions(_) for _ in node['code']]
            blocks[start] = CodeBlock(start, code)

        # then filling them out
        for start in blocks:
            block = blocks[start]

            block.ins = list(map(
                lambda _: blocks[_],
                nodes[start]['ins'],
            ))

            block.outs = list(map(
                lambda _: blocks[_],
                nodes[start]['outs'],
            ))

        self.logger.debug(blocks)

        return blocks[address_start]

    def variables(self) -> List[Variable]:
        return [deserialize_expressions(_) for _ in self.json['vars']]


class CFGCodeNode(CFGNode, Protocol):
    """Prototype for the nodes of the structured CFG."""
    ins: List[CFGCodeNode]
    outs: List[CFGCodeNode]
    # change this to overrides, for a while body this is "break"
    # maybe create a class Override with a type GOTO or BREAK
    # this obviously has to generate custom expressions
    # to indicate in the CodeInterpreter the right statements
    gotos: List[CFGCodeNode]


class CFG(LoggerMixin):
    """Wrap the ghidra's BasicBlock CFG in order to do operations."""

    def __init__(self, starting_block: CodeBlock):
        self._blocks = starting_block

    def get_head(self) -> CodeBlock:
        return self._blocks

    def traverse(self, entry: CodeBlock = None) -> Generator[CodeBlock, None, None]:
        entry = self._blocks if entry is None else entry

        for kind, block in traverse(entry):
            if kind != NodeType.NEW:
                continue

            yield block

    def json(self):
        """Return a serialized dump of the CFG"""
        cfg = {}
        for block in self.traverse():
            cfg[block.start] = {
                'ins' : list(map(operator.attrgetter('start'), block.ins)),
                'outs': list(map(operator.attrgetter('start'), block.outs)),
                'code': list(map(serialize_expression, block.ops)),
            }

        return cfg


class Function(LoggerMixin):
    """Wrap the representation of a function.

    In ghidra there are two main representations, Function and HighFunction."""

    def __init__(self, provider: Provider):
        self.provider = provider

        self.cfg = CFG(self.provider.cfg())
        self.variables = self.provider.variables()

    def json(self) -> Dict:
        return {
            'vars' : list(map(serialize_expression, self.variables)),
            'start': self.provider.get_start(),
            'cfg'  : self.cfg.json(),
        }

    def get_structured(self) -> CFGCodeNode:
        pass


class CodeComponent(Protocol):
    """Main protocol to describe a code component.

    A code component is an abstraction of the CFG that groups node relations
    to form high-level language constructs; in our language of choice, that
    tries to be as near as C as possible, the grammar is the following

        S ::= x = A
          | ;
          | if (B) S
          | if (B) S else S
          | while (B) S
          | break;
          | { Sl }
        Sl ::= Sl | S | €
        P  ::= Sl

    Each one can be translated in a unique way to some binary tree configuration;
    the basic case are instructions in the same block that can be safely grouped
    into compound statements.

    The real interesting nodes are the nodes with two outgoing edges, those can be
    the head of "while" or "if" statements. The nodes belonging to the first case
    are easily detectable since have paths "uniquely" coming back to such nodes.

    The "break" statement is strictly linked to such nodes because it's the action
    that sends the execution path immediately to the exit node of the immediate
    surrounding loop. Note that this statement will be inside an "if" otherwise
    would short circuit the loop, removing it altogether. Since we are trying to
    reconstruct from compiled code, accessing only the final CFG, it's impossible
    in such case obtain a fragment of code like

        while (...) {
            ...
            break;
            ...
        }

    Now, for the "if" component, we have the following cases

     - one branch (the true one)
     - both branches (with the "else")
     - the variant where the "if" contains a "break"
     - a branch contains at most a "break" (degenerate)
"""

    @classmethod
    def detect(cls, node: CFGCodeNode, loops: CFGLoops) -> Optional[CFGCodeNode]:
        pass


logger_components = logging.getLogger("{}.{}".format(__name__, "components"))


class Label:
    """Represent a program point"""

    def __init__(self, index: int, op: Op):
        self.index = index
        self.op = op

    @staticmethod
    def generator() -> Generator[Label, Op, None]:
        def _coroutine():
            index = 0

            op = None

            while True:
                if not op:
                    op = yield
                    continue
                index += 1
                op = yield Label(index, op)

        g = _coroutine()
        # you need to call send() with a None argument as a first thing
        # https://peps.python.org/pep-0342/#new-generator-method-send-value
        g.send(None)

        return g


# TODO: factorize code __init__() (initializing gotos for example)
class BaseBlock(ABC, CFGCodeNode):

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

    def structure_gotos(self, src: List[CFGCodeNode]) -> List[CFGCodeNode]:
        return list(itertools.chain.from_iterable(map(
            operator.attrgetter('gotos'),
            src,
        )))

    @property
    def n_ins(self):
        return len(self.ins)

    @property
    def n_outs(self):
        return len(self.outs)

    @abstractmethod
    def at(self) -> Label:
        """Program point at which execution of this program component starts"""

    @abstractmethod
    def after(self) -> Label:
        """Program exit point after this component"""

    @abstractmethod
    def escape(self) -> bool:
        """Indicate wheter or not this program component contains a break"""

    @abstractmethod
    def break_to(self) -> Label:
        """Program point destination of a "break"."""

    @abstractmethod
    def break_of(self) -> Set[Label]:
        """Set of labels where the "break" statement of this components are."""

    @abstractmethod
    def into(self) -> Set[Label]:
        """Set of program points inside this program component (excluding `after()`
        and `break_to()`)."""

    @abstractmethod
    def labs(self) -> Set[Label]:
        """Potentially reachable program points while and after executing this program component"""

    @abstractmethod
    def labx(self) -> Set[Label]:
        """Potentially reachable program points while, after and as a consequence of a break."""


class SimpleBaseBlockInsMixin:
    def helper_structure_outs_for_ins(self: CFGNode, head: CFGNode, ins: List[CFGNode]):
        """Modify the exiting edges from the entering side in order to swap
        the old `head` with the instance itself."""
        for entering in ins:
            outs = entering.outs[:]
            for index, out in enumerate(outs):
                if out == head:
                    entering.outs[index] = self
                    break


class SimpleBaseBlockOutsMixin:
    def helper_structure_ins_for_outs(self: CFGNode, head: CFGNode, outs: List[CFGNode]):
        """Modify the entering edges from the exiting side in order to swap
        the old `head` with the instance itself."""
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

    def __init__(self, head: CFGCodeNode, body: CFGCodeNode, out: CFGCodeNode):
        self._head = head
        self._body = body
        self.start = self._head.start

        # now we want to reconstruct the ins/outs
        # the ins are the ins of the head but with the edge from the body removed
        self.ins = list(set(self._head.ins) - {self._body})
        self.structure_outs_for_ins(self.ins)

        self.outs = [out]
        self.structure_ins_for_outs(self.outs)

        # self.gotos = self.structure_gotos([self._body])

    def __repr__(self):
        return "<{}(head=@{:x}, exit=@{:x})>".format(
            self.__class__.__name__,
            self._head.start,
            self.outs[0].start,
        )

    def __contains__(self, item):
        return item is self._head or item in self._body

    def at(self) -> Label:
        pass

    def after(self) -> Label:
        pass

    def escape(self) -> bool:
        pass

    def break_to(self) -> Label:
        pass

    def break_of(self) -> Set[Label]:
        pass

    def into(self) -> Set[Label]:
        pass

    def labs(self) -> Set[Label]:
        pass

    def labx(self) -> Set[Label]:
        pass

    def structure_ins_for_outs(self, outs: List[CFGNode]):
        self.helper_structure_ins_for_outs(self._head, outs)

    def structure_outs_for_ins(self, ins: List[CFGNode]):
        self.helper_structure_outs_for_ins(self._head, ins)

    def is_head(self, block):
        return block is self._head

    @property
    def head(self):
        return self._head

    def code(self, indent=0):
        head = ["while (", *self._head.code(indent=4), ") {"]
        body = self._body.code(indent=4)
        tail = ["}"]

        return list(map(
            lambda _: "{}{}".format(" " * indent, _),
            itertools.chain.from_iterable([head, body, tail]),
        ))

    @classmethod
    def detect(cls, node: CFGNode, loops: CFGLoops) -> Optional[WhileBlock]:
        # first of all check that is the head of a loop and that is a branching node
        if node.n_outs != 2 or (node not in [_.head for _ in loops]):
            return None

        # loop over the combinations of the exit nodes
        # TODO: save the fact that the body is the false branch
        for body, exit in permutations(node.outs, 2):
            if body.n_outs != 1:  # the body has one exit node
                continue

            if body.n_ins != 1:  # the body must have only the head pointing to it
                continue

            if body.outs[0] != node:  # the body come back to the head
                continue

            logger_components.debug("detected WhileBlock at @%x", node.start)
            return cls(node, body, exit)

        return None


class IfBlock(BaseBlock):

    def at(self) -> Label:
        pass

    def after(self) -> Label:
        pass

    def escape(self) -> bool:
        pass

    def break_to(self) -> Label:
        pass

    def break_of(self) -> Set[Label]:
        pass

    def into(self) -> Set[Label]:
        pass

    def labs(self) -> Set[Label]:
        pass

    def labx(self) -> Set[Label]:
        pass

    def __init__(self, head: CFGCodeNode, left: CFGCodeNode, right: Optional[CFGCodeNode], out: Optional[CFGCodeNode]):
        self._head = head
        self._body = [left, right]
        self.start = self._head.start

        self.outs = [out] if out else []

        self.ins = self._head.ins

        # fix the ins and outs
        if self.outs:
            self.structure_ins_for_outs(self.outs)
        self.structure_outs_for_ins(self.ins)

        # remove None if present
        body = list(filter(bool, self._body))

        # self.gotos: List[CFGNode] = self.structure_gotos(body)

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
        head = ["if (", *self._head.code(), ") {"]
        true = self._body[0].code(indent=4)
        false = itertools.chain.from_iterable([["} else {"], self._body[1].code(indent=4)]) if self._body[1] else []
        tail = ["}"]

        return list(map(
            lambda _: "{}{}".format(" " * indent, _),
            itertools.chain.from_iterable([head, true, false, tail]),
        ))

    @classmethod
    def detect(cls, node: CFGCodeNode, loops: CFGLoops) -> Optional[IfBlock]:
        # first of all we need a branch that is not a loop
        if node.n_outs != 2 or node in [_.head for _ in loops]:
            return None

        # loop over the combinations of the exit nodes
        for a, b in permutations(node.outs, 2):
            # we consider b the true condition

            # if b has not exit
            if b.n_outs == 0:
                logger_components.debug("detected IfBlock with a dead end (goto?) for %s", node)
                # then has an else without exit
                false = a if a.n_outs == 0 else None
                # and has not exit node
                out = a if false is None else None

                return cls(node, b, false, out)

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


class SkipBlock:
    """Simple block for empty statements list."""

    def __init__(self, block: CFGCodeNode, entering: CFGCodeNode, exiting: CFGCodeNode):
        self._block = block
        self.ins = [entering]
        self.outs = [exiting]


class BreakBlock(BaseBlock, SimpleBaseBlockInsMixin):
    """Empty block representing a break statement"""

    def __init__(self, entering: CFGCodeNode, out: CFGCodeNode):
        """Setup the block to point (via a "break") to `out`.

        This block is used to substitute the edge between an internal node
        of a while loop's body and the exit node.

        In this configuration, ``"""
        self.start = random.randint(0, 2**32)  # TODO: workaround
        self.breaks_to = out  # FIXME: naming
        self.ins = [entering]
        self.outs = []

        self.structure_outs_for_ins(self.ins)
        self.structure_ins_for_outs([out])

    def structure_ins_for_outs(self, outs: List[CFGNode]):
        """We want to remove the entering element from the ins of the exiting block"""
        out = outs[0]  # we should have only one exiting edge

        out.ins.remove(self.ins[0])  # remove

    def structure_outs_for_ins(self, ins: List[CFGNode]):
        """Put this block in between the entering and exiting node."""
        outs = self.ins[0].outs
        for idx, out in enumerate(outs):
            if out == self.breaks_to:
                outs[idx] = self

    def at(self) -> Label:
        pass

    def after(self) -> Label:
        pass

    def escape(self) -> bool:
        pass

    def break_to(self) -> Label:
        pass

    def break_of(self) -> Set[Label]:
        pass

    def into(self) -> Set[Label]:
        pass

    def labs(self) -> Set[Label]:
        pass

    def labx(self) -> Set[Label]:
        pass


class ContiguousBlock(BaseBlock, SimpleBaseBlockInsMixin, SimpleBaseBlockOutsMixin):
    """Aggregate together blocks that are contiguous.

    In practice is the CFG representation of the grammar rule

        Sl ::= Sl S


    """

    def __init__(self, head: ContiguousBlock | CFGNode, statement: CFGNode):
        self.sl = head
        self.s = statement
        self.start: int = self.sl.start

        # self.gotos = self.structure_gotos([self._blocks, self._blocks.outs[0]])

        self.outs = self.s.outs
        self.ins = self.sl.ins

        self.structure_outs_for_ins(self.ins)

        if self.outs:
            self.structure_ins_for_outs(self.outs)

    def structure_ins_for_outs(self, outs: List[CFGNode]):
        self.helper_structure_ins_for_outs(self.s, outs)

    def structure_outs_for_ins(self, ins: List[CFGNode]):
        self.helper_structure_outs_for_ins(self.sl, ins)

    def __repr__(self):
        return "<{}(@{:x})>".format(self.__class__.__name__, self.start)

    def at(self) -> Label:
        pass

    def after(self) -> Label:
        pass

    def escape(self) -> bool:
        pass

    def break_to(self) -> Label:
        pass

    def break_of(self) -> Set[Label]:
        pass

    def into(self) -> Set[Label]:
        pass

    def labs(self) -> Set[Label]:
        pass

    def labx(self) -> Set[Label]:
        pass

    def code(self, indent=0):
        return list(map(
            lambda _: "{}{}".format(" " * indent, _),
            self._blocks.code() + self._blocks.outs[0].code(),
        ))

    @classmethod
    def detect(cls, node: CFGCodeNode, loops: CFGLoops) -> Optional[ContiguousBlock]:
        # we must have only one exit edge
        if node.n_outs != 1:
            return None

        out = node.outs[0]

        # this edge must have us as only predecessor
        if out.n_ins != 1:
            return None

        # then we can unite them together
        logger_components.debug("detected ContiguousBlock @%s", repr(node))
        return cls(node, out)


class CodeBlock(BaseBlock, LoggerMixin):
    """Wrap the ghidra's blocks in order to allow for reworking of the CFG
    with higher-level abstraction."""

    def structure_ins_for_outs(self, outs: List[CFGNode]):
        pass

    def structure_outs_for_ins(self, ins: List[CFGNode]):
        pass

    def at(self) -> Label:
        return self.ops[0]

    def after(self) -> Label:
        raise NotImplementedError()

    def escape(self) -> bool:
        raise NotImplementedError()

    def break_to(self) -> Label:
        raise NotImplementedError()

    def break_of(self) -> Set[Label]:
        raise NotImplementedError()

    def into(self) -> Set[Label]:
        return set(self.ops)

    def labs(self) -> Set[Label]:
        raise NotImplementedError()

    def labx(self) -> Set[Label]:
        raise NotImplementedError()

    def __init__(self, start: int, labels: List[Expression], ins: List[CFGNode] = None, outs: List[CFGNode] = None):
        self.ops = labels
        self.start = start
        self.ins = ins
        self.outs = outs

        self.gotos: List[CFGNode] = []  # a CodeBlock can have only one goto but in general more are possible

    def __repr__(self):
        return "<{}(@{:x})>".format(self.__class__.__name__, self.start)

    @property
    def n_ins(self):
        return len(self.ins)

    @property
    def n_outs(self):
        return len(self.outs)


class StructureCode(LoggerMixin):
    """Main component used to restructure the CFG to code blocks.
    
    The algorithm has the following iterated steps:
    
     1. merge the contiguous blocks together in order to obtain a "reduced graph"
     2. meanwhile build a list of all the paths buildable into this graph
        allowing to categorize all the nodes in three types
        
         a. while's header (where a loop starts)
         b. if's header (node with branching)
         c. normal block (not included in the previous twos)

     3. detect the "break" statements as the edges that jump from the
        body of a loop to its exit node and put in place a suited block
        to represent it
     4. iterate on each node trying to see if the pattern of a specific
        code component fits and in case put it in place

    This four steps are iterated until no blocks are replaced in the process
    and the final number of blocks remained is equal to one; it it's not the
    case then tries to detect "goto" statements and removes edges in order to
    reduce the graph "complexity" and repeat the process.
    """
    COMPONENTS: List[CodeComponent] = [
        # ContiguousBlock,
        IfBlock,
        WhileBlock,
    ]

    def __init__(self, cfg: CFGCodeNode):
        self._head = cfg
        # save here non local information useful
        # for structuring the code
        self.paths: CFGNodeToPaths = {}
        self.nodes: CFGPathToNodes = {}
        self.loops: List[CFGLoop] = []

    def get(self) -> CFGCodeNode:
        return self._head

    def reset_loops_traversing(self) -> Dict[int, bool]:
        # create a dictionary with the starting address of the block
        # containing the head of the loop indicating if we have traversed
        # such block already
        loops_traversed = {}
        for loop in self.loops:
            loops_traversed[loop.head.start] = False

        return loops_traversed

    def do_linearize(self):
        changed = True

        while changed:
            for changed, new_head in self.iter_linearize():
                self.logger.debug("do_linearize iteration")
                self._head = new_head

    def iter_linearize(self) -> Generator[Tuple[bool, CFGNode], None, None]:
        changed, new_head = True, self._head

        self.paths, self.nodes, self.loops = explore_all(self._head)

        self.logger.debug("iter_linearize: nodes: %s", self.nodes)

        _, new_head = self.merge_contiguous_blocks(new_head, self.reset_loops_traversing())

        self.detect_break()

        while changed:
            changed_by_merge, new_head = self.merge_contiguous_blocks(new_head, self.reset_loops_traversing())
            changed_by_linearization, new_head = self.linearize(new_head, dict(), self.reset_loops_traversing())

            changed = changed_by_merge or changed_by_linearization

            self.logger.debug("new head: %s, changed: %s", repr(new_head), changed)
            yield changed, new_head

    def detect_break(self):
        """Do what is necessary in order to setup the "break" statements."""
        # take the loops
        for loop in self.loops:
            # take the exit node
            exit = loop.exit
            # take the paths the cross such node
            paths = self.paths[exit]

            for path in paths:
                # then detect which path comes from this loop
                if path.startswith(loop.path):
                    # remove the last branching to have the
                    # path of the if
                    path_from_break, direction = path[:-1], path[-1]
                    nodes_w_break = self.nodes[path_from_break]

                    # for each node we found, put a BreakNode to cut the edge
                    for n_w_break in nodes_w_break:
                        self.logger.debug(
                            "found a 'break' at '%s' with direction '%s' and exit '%s'",
                            n_w_break, direction, exit
                        )
                        bb = BreakBlock(n_w_break, exit)

    def detect_gotos(self) -> None:
        """Try to find the most plausible gotos.

        The logic here is that, after doing the exploration via binary paths,
        since we have a "irreducible graph" without concatenable blocks, probably
        we have to add some gotos and we chose the gotos trying to linearize
        the longest path removing the incoming edges from it."""
        # find the path with the most occurrences
        count: Dict[str, Tuple[int, int, int]] = {}

        for path in self.nodes:
            count[path] = (len(self.nodes[path]), len(path), len(path.replace("1", "")))

        # sort with respect to
        #  1. number of nodes (create more concatenable paths)
        #  2. length of the path (less digits means less nested conditions)
        #  3. number of zeroes (more zeroes means more default if condition)
        first_stage = sorted(count.items(), key=lambda _: _[1][0], reverse=True)

        logger_components.debug("try to linearize this: %s", first_stage)

        # this is the path we want to straighten
        winner_path, _ = first_stage[0]

        print(winner_path)
        # get the nodes where there are other paths incoming
        # other than the winner_path
        winner_nodes = self.nodes[winner_path]
        print(winner_nodes)

        # now we can traverse the nodes with the given path
        # since by the assumptions we have a certain numbers
        # of nodes in a row without branching
        node = winner_nodes[0]  # the first entry is the first node with such path
        previous = None

        while node in winner_nodes:
            print(node)
            if previous is not None:
                extra = set(node.ins) - {previous}

                for entering in extra:
                    self.logger.info("goto: %s <- %s", node, entering)
                    node.ins.remove(entering)
                    entering.outs.remove(node)
                    entering.gotos.append(node)

                    print("entering outs':", entering.outs)

            if node.n_outs == 0:
                break

            previous = node
            node = node.outs[0]

    def merge_contiguous_blocks(self, head: CFGNode, loops_traversed: Dict[int, bool]) -> Tuple[bool, CFGNode]:
        """Rework the graph in order to consolidate contiguous blocks together and
        return the new head"""
        self.logger.debug("try merging block %s", head)
        # first of all check that we are returning on a node
        if head.start in loops_traversed:
            traversed = loops_traversed[head.start]

            if traversed:
                return False, head

            loops_traversed[head.start] = True

        node = head

        # this first loop tries to merge the node starting from the head
        # we want to continue until there are no more possible merging
        while (last_node := ContiguousBlock.detect(node, self.loops)) is not None:
            self.logger.debug("merged head in place %s -> %s", node, last_node)
            node = last_node

        # if something happened to the head, it's here
        new_head = node
        has_changed = new_head != head

        # now recursively act on its child
        for index in range(node.n_outs):
            child = node.outs[index]

            child_changed, _ = self.merge_contiguous_blocks(child, loops_traversed)

            if child_changed:
                has_changed = True

        return has_changed, new_head

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
            found = component.detect(node, self.loops)

            if found:
                return True, found

        return changed, node
