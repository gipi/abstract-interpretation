from __future__ import annotations

import logging
from enum import Enum, auto
from typing import TypeVar, Tuple, Generator


class LoggerMixin:
    @property
    def logger(self):
        if not hasattr(self, '_logger') or not self._logger:
            self._logger = logging.getLogger(
                "{}.{}".format(self.__module__, self.__class__.__name__)
            )

        return self._logger


class NodeType(Enum):
    NEW = auto()
    OLD = auto()
    ANCESTOR = auto()


T = TypeVar("T")


def traverse(node: T, visited=None, ancestors=None, childs = lambda _: _.outs) -> Generator[Tuple[NodeType, T], None, None]:
    """Traverse a "generic" directed graph.

    We want something interesting from this traversal algorithm: we want
    to know if a child is an already encountered one and also if it's one
    of our ancestors (so that we know we are element of a loop).

    The basic block from ghidra have at most two child, creating a binary tree
    of sort (when a child is present is always left(?) and a child can point "backward");
    we don't want to continue if the child is a node already encountered
    but we want to know if is one of our ancestors.

    From a basic layout like the following

              <A>
             /   \
           <B>   <C>
          /   \
        <D>   <E>

    we want to ordering (A, B, D, E, C) that is the pre-order traversal algorithm.

    In our case one of the leaf node might be an already encountered node.

    The elements in ancestors are the path to reach the current element.
    """
    # root node (and add to the ancestors)
    # left node
    # right node
    # at the end remove the root node from the ancestors

    # trivial case
    if node is None:
        return

    visited = visited if visited is not None else {}
    ancestors = ancestors if ancestors is not None else []

    # first check is not an ancestor
    if node in ancestors:
        yield NodeType.ANCESTOR, node
        return

    # maybe already encountered
    if node in visited:
        yield NodeType.OLD, node
        return

    visited[node] = True
    ancestors.append(node)

    yield NodeType.NEW, node

    for child in childs(node):
        yield from traverse(child, visited=visited, ancestors=ancestors, childs=childs)

    ancestors.pop()


