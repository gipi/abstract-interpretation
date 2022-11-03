import logging
from enum import Enum, auto

from ghidra.program.model.address import AddressSpace
from ghidra.program.model.pcode import HighConstant, HighLocal, HighOther, HighParam, HighVariable, Varnode


logger = logging.getLogger(__name__)


class Op:
    def __init__(self, pcode):
        self._pcode = pcode
        self.inputs = [Variable(i) for i in self._pcode.getInputs()]

        output = self._pcode.getOutput()

        self.output = output if not output else Variable(output)

    def __str__(self):
        output = "{} = ".format(self.output.name()) if self.output else ""

        return "{}{} {}".format(
            output,
            self._pcode.getMnemonic(),
            ", ".join([str(_.name()) for _ in self.inputs]),
        )


class VariableType(Enum):
    CONSTANT = auto()
    LOCAL = auto()
    OTHER = auto()
    PARAM = auto()

    @classmethod
    def map(cls, high_variable: HighVariable):
        # here I would have liked to have directly the class
        # but ghidra_bridge wraps all of them so it's difficult to to
        t = type(high_variable).__name__.split(".")[-1]
        return {
            'HighConstant': cls.CONSTANT,
            'HighLocal':    cls.LOCAL,
            'HighOther':    cls.OTHER,
            'HighParam':    cls.PARAM,
        }[t]


class Variable:
    """Wrap the varnode data type in order to be able
    to be manageable."""
    def __init__(self, varnode: Varnode):
        logger.debug("%s initializing: %s (%s)", self.__class__, varnode, varnode.getHigh())
        self._varnode = varnode

        # there is getSpace() but it's spaceID that
        # I don't know what is
        self.space = self._varnode.getAddress().getAddressSpace().getType()
        self.offset = self._varnode.getOffset()
        self.size = self._varnode.getSize()

        self.type: VariableType = VariableType.map(self._varnode.getHigh()) if varnode.getHigh() else None

    def __repr__(self):
        return "<{}({} {},{})>".format(
            self.__class__,
            self.type,
            self.address_space(),
            self.offset,
        )

    def address_space(self) -> str:
        if self.space == AddressSpace.TYPE_CONSTANT:
            return 'const'
        elif self.space == AddressSpace.TYPE_UNIQUE:
            return 'unique'
        elif self.space == AddressSpace.TYPE_STACK:
            return 'stack'
        elif self.space == AddressSpace.TYPE_RAM:
            return 'ram'
        elif self.space == AddressSpace.TYPE_CODE:
            return 'code'
        elif self.space == AddressSpace.TYPE_REGISTER:
            return 'register'
        else:
            return 'unknown_{:d}'.format(self.space)

    def name(self) -> str:
        """This is true if this variable is a HighVariable"""
        high = self._varnode.getHigh()

        if high and high.getName() and high.getName() != "UNNAMED":
            return high.getName()

        addr_space = self.address_space()

        if addr_space:
            return "{}_{:x}".format(addr_space, self.offset)

        return None

    def __members(self):
        return self.space, self.offset, self.size

    def __eq__(self, other):
        if type(self) == type(other):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())
