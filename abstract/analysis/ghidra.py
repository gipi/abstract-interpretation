"""
Interface between high level analysis and the data types
provided by Ghidra itself.

Roughly speaking, Ghidra provides two objects to interact with the analysis
the Pcodes (that you can interpret as the operations/instructions) and the
Varnodes (variables and data).

The problem with all of this is that these two entities are pretty strictly
tied to implementation detail of Ghidra itself, we want only to obtain a high
level and structured interpretation of the code so we want to abstract away
as much as possible.
"""
import ghidra_bridge

bridge = ghidra_bridge.GhidraBridge(
    namespace=globals(),
    hook_import=True,  # this allows to import "ghidra"'s packages
)
from enum import Enum, auto
from typing import Dict, Tuple, Optional, List, TypedDict

from ghidra.program.model.address import AddressSpace as GhidraAddressSpace
from ghidra.app.decompiler import DecompileOptions, DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

from .code import CodeBlock, Provider
from .graph import BasicBlock
from ..pcode import (
    Variable,
    Type,
    Constant,
    BinaryExpression,
    UnaryExpression,
    Operator,
    FunctionRef,
    Expression,
    FunctionCall,
    NopExpression,
    FieldVariable,
    ControlFlowExpression,
    ComponentType,
)
from ..utils import LoggerMixin, traverse, NodeType


def decompile(func, program=None):
    monitor = ConsoleTaskMonitor()
    ifc = DecompInterface()
    options = DecompileOptions()
    ifc.setOptions(options)
    program = program or currentProgram
    ifc.openProgram(program)
    res = ifc.decompileFunction(func, 60, monitor)
    return res


class AddressSpace(Enum):
    """Wrap Ghidra's address spaces"""
    CONSTANT = GhidraAddressSpace.TYPE_CONSTANT
    UNIQUE = GhidraAddressSpace.TYPE_UNIQUE
    STACK = GhidraAddressSpace.TYPE_STACK
    RAM = GhidraAddressSpace.TYPE_RAM
    CODE = GhidraAddressSpace.TYPE_CODE
    REGISTER = GhidraAddressSpace.TYPE_REGISTER


class Opcode(Enum):
    """Enumeration for the Pcode's opcode.

    Its primarly usage is to match the mnemonic."""
    CALL = auto()  # TODO: this needs to retrieve the actual function from the address
    BRANCH = auto()
    CBRANCH = auto()
    CAST = auto()
    INT_EQUAL = auto()
    INT_NOTEQUAL = auto()
    INT_LEFT = auto()
    INT_SLESS = auto()
    INT_ADD = auto()
    INT_SDIV = auto()
    INT_MULT = auto()
    INT_AND = auto()
    INT_SEXT = auto()
    INT_ZEXT = auto()
    COPY = auto()
    STORE = auto()
    LOAD = auto()
    MULTIEQUAL = auto()
    INDIRECT = auto()
    SUBPIECE = auto()
    PTRADD = auto()
    PTRSUB = auto()
    RETURN = auto()


"""
StackRef and Parameter shouldn't be used out of this module because
are an implementation detail of the translation process.
"""


class Ram:
    """Sentinel value for a ram address"""

    def __init__(self, address):
        self.address = address


class StackRef:
    """It's simply a sentinel value to indicate the stack pointer"""


class Parameter:
    """It's simply a sentinel value for internal parametrization of Ghidra."""


class PartialFieldVariable(FieldVariable):
    """Represents a Field variable that is unresolved."""

    def __init__(self, parent: Variable, name: str, offset: int):
        super().__init__(parent, name)

        self.offset = offset

    def __repr__(self):
        return "<{}(parent={}, name={}, offset=0x{:x})>".format(
            self.__class__,
            self.parent,
            self.name,
            self.offset,
        )

    def __str__(self):
        return "{}.{} + 0x{:x}".format(
            self.parent.name,
            self.name,
            self.offset,
        )


class Branch:
    """Sentinel value to indicate a unconditional branch"""

    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return "<{}(target={})>".format(self.__class__.__name__, self.target)


# TODO: this should be factorized in some abstract class that from a
#       particular intermediate language extracts all the information
#       from a function
class GhidraProvider(Provider, LoggerMixin):
    """Encapsulates all the information provided by the Ghidra's decompiler
    related to a given function.

    By this class you can obtain the local variables, parameters and control flow graph."""

    def get_start(self) -> int:
        return self.context.getFunction().getEntryPoint().getOffset()

    def __init__(self, function):
        self.context = function

        self._registry_type: Dict[str, Type] = {}
        self._registry_unnamed: Dict[Tuple[AddressSpace, int, int], Variable] = {}
        self._registry_named: Dict[str, Variable] = {}

        self._index = 0

    @staticmethod
    def get_stack_pointer():
        return currentProgram.getLanguage().getDefaultCompilerSpec().getStackPointer()

    def _get_next_available_name(self) -> str:
        """Generate a name for a intermediate variable"""
        new_name = "tmp{}".format(self._index)

        self._index += 1

        return new_name

    def _normalize_stack_offset(self, offset: int) -> int:
        mask = 0xffffffffffffffff  # TODO: generalize width
        return -((~offset & mask) + 1)

    def resolve_variable_by_stack_offset(self, offset: int) -> 'GhidraVariable':
        return self.context.getFunction().getStackFrame().getVariableContaining(self._normalize_stack_offset(offset))

    def resolve_stack_variable(self, offset, size) -> Variable | FieldVariable:
        # check there is a variable at that offset
        # NOTE: this is a ghidra's Variable class
        stack_variable = self.resolve_variable_by_stack_offset(offset)

        # it's possible that the data type is composite so the offset
        # it's in reality a base (the stack_variable) plus the internal offset
        # to the internal member of this composite data type
        base = stack_variable.getStackOffset()
        internal_offset = offset - base
        data_type = self._resolve_datatype(stack_variable.getDataType())
        name = stack_variable.getName()

        # for now create the variable
        variable = Variable(name, data_type)

        # if it's not composite then our job here is done
        if not data_type.is_composite:
            return variable

        # note here that the _field_ has a matching data type with the varnode
        # but the data type of the stack_variable is the composite type
        # containing the varnode itself
        # stack_data_type = self._registry_type[stack_variable.getDataType()]

        # it's also possible that we are looking directly at the composite type
        # if stack_data_type == data_type:
        #    return Variable(name, data_type)

        # here instead we have an internal component
        found_offset, name_field, type_field = data_type.get_field_at_offset(internal_offset)

        if found_offset != internal_offset:
            self.logger.warning("found offset doesn't match requested one")
            return PartialFieldVariable(
                variable,
                name_field,
                internal_offset - found_offset,
            )

        return FieldVariable(
            variable,
            name_field,
        )

    def _resolve_datatype(self, g_data_type) -> Type:
        """Recursively tries to resolve in our own data type"""
        # we have cached it previously
        name = g_data_type.getName()

        if not name:
            raise ValueError("{} has not name".format(g_data_type))

        if name in self._registry_type:
            return self._registry_type[name]

        points_to = None

        # special case for PointerDataType
        if hasattr(g_data_type, 'newPointer'):
            # we save also the original data type
            g_deref_data_type = g_data_type.getDataType()
            points_to = self._resolve_datatype(g_deref_data_type)

        # we need to create ex-novo, adding possibly all the needed sub-type
        # if it's composed
        if hasattr(g_data_type, 'getNumComponents'):
            num = g_data_type.getNumComponents()  # be aware of getNumDefinedComponents()

            components = []

            for idx in range(num):
                component = g_data_type.getComponent(idx)

                offset: int = component.getOffset()  # offset in the struct
                sub_ghidra_data_type = component.getDataType()  # data type

                # translate in my own data type
                sub_data_type = self._resolve_datatype(sub_ghidra_data_type)

                components.append(ComponentType(offset, component.getFieldName(), sub_data_type))

            self._registry_type[name] = Type(name, g_data_type.getLength(), components)
            return self._registry_type[name]

        # if we are here we have a plain data type
        self._registry_type[name] = Type(name, g_data_type.getLength(), [], points_to=points_to)
        return self._registry_type[name]

    def variable(self, varnode) -> Variable | FieldVariable | FunctionRef | StackRef | Constant | Parameter | Ram:
        # ghidra's varnodes are at the lowest level
        # simply a tuple of three values
        addr_space = AddressSpace(varnode.getAddress().getAddressSpace().getType())
        offset = varnode.getOffset()
        size = varnode.getSize()

        triple = addr_space, offset, size

        # first of all see if it's a high variable
        high = varnode.getHigh()

        if not high:
            if addr_space == AddressSpace.RAM:
                # we suppose the ram address is indicating a function
                # otherwise a variable would be attached to it
                ghidra_function = getFunctionAt(varnode.getAddress())

                # if it's not a function one day we'll understand
                # the edge cases, for now return a Ram object
                if not ghidra_function:
                    return Ram(varnode.getAddress())

                signature = [self._resolve_datatype(_.getDataType()) for _ in
                             ghidra_function.getSignature().getArguments()]

                kwargs = {}

                if output := ghidra_function.getReturnType():
                    kwargs['return_type'] = output

                return FunctionRef(
                    ghidra_function.getName(),
                    *signature,
                    **kwargs,
                )

            # 3. it's a variable located into the stack
            if addr_space == AddressSpace.STACK:
                return self.resolve_stack_variable(offset, size)
            # if reach here probably it's something related to internal
            # parametrization of Ghidra's Pcode operations (like the second
            # argument of INDIRECT)
            if addr_space == AddressSpace.CONSTANT:
                return Parameter()

            raise ValueError("{} has not High representation".format(varnode))

        # now we have high variable
        ghidra_data_type = high.getDataType()

        # TODO: maybe saving the triple and then on retrieving checking
        #       that the datatype corresponds
        data_type = self._resolve_datatype(ghidra_data_type)

        # if the triple is present in the cache then return it
        if triple in self._registry_unnamed:
            return self._registry_unnamed[triple]

        # if it's a constant then return a constant
        # TODO: if it's an unnamed constant?
        if addr_space == AddressSpace.CONSTANT:
            return Constant(offset, data_type)

        # let's see if we are able to resolve the variable
        # 1. by name (be aware that HighConstant has getName() returning None)
        if (name := high.getName()) != 'UNNAMED' and name:
            if name in self._registry_named:
                return self._registry_named[name]

            variable = Variable(name, data_type)

            self._registry_named[name] = variable

            return variable

        if (symbol := high.getSymbol()) and (name := symbol.getName()):
            if name in self._registry_named:
                return self._registry_named[name]

            variable = Variable(name, data_type)

            self._registry_named[name] = variable

            return variable

        # the last chance that is something interesting is
        # 2. by the stack register
        if addr_space == AddressSpace.REGISTER and (offset == self.get_stack_pointer().getOffset()):
            return StackRef()  # this MUST be resolved otherwise you have a dangling pointer to unknown

        if addr_space == AddressSpace.STACK:
            return self.resolve_stack_variable(offset, size)

        # if nothing works, then assign a name and save it
        name = self._get_next_available_name()
        variable = Variable(name, data_type)

        self._registry_unnamed[triple] = variable  # TODO: maybe save the triple -> name association

        return variable

    def get_varnode_triple(self, varnode) -> Tuple[AddressSpace, int, int]:
        """Return the AddressSpace, offset and size tuple."""
        addr_space = AddressSpace(varnode.getAddress().getAddressSpace().getType())
        offset = varnode.getOffset()
        size = varnode.getSize()

        return addr_space, offset, size

    def is_varnode_terminal(self, varnode) -> bool:
        """Determines if the varnode passed as argument is a terminal
        in the implicit AST of an expression."""
        self.logger.info("is terminal %s", varnode)

        # here we are practically looking for a variable
        # recognized by Ghidra
        high = varnode.getHigh()

        addr_space, offset, size = self.get_varnode_triple(varnode)

        if not high:
            # an address in the RAM address space
            if addr_space == AddressSpace.RAM:
                self.logger.info("< RAM")
                return True

            if addr_space == AddressSpace.CONSTANT:
                self.logger.info("< CONSTANT")
                return True

            self.logger.info("< not terminal (without variable and not RAM)")
            return False

        # if we have a named variable then go for it
        # be aware that HighConstant have getName() returning None :)
        if (name := high.getName()) != 'UNNAMED' and name:
            self.logger.info("< named variable ('%s')", name)
            return True

        if (symbol := high.getSymbol()) and (name := symbol.getName()):
            self.logger.info("< named symbol ('%s')", name)

            if symbol.isParameter():
                offset = high.getOffset()
                self.logger.info("<  this has offset %d", offset)

                if offset != -1:
                    self.logger.info("< storage doesn't match")

            return True

        # the only cases where an unnamed variable is terminal
        # it's when it's in the stack
        if addr_space == AddressSpace.STACK:
            self.logger.info("< STACK")
            return True

        # or it's the stack register itself
        if addr_space == AddressSpace.REGISTER and (offset == self.get_stack_pointer().getOffset()):
            self.logger.info("< STACK POINTER")
            return True

        # or simply an unnamed constant
        if addr_space == AddressSpace.CONSTANT:
            self.logger.info("< CONSTANT")
            return True

        self.logger.info("< not terminal")
        return False

    def translate_block(self, basic_block) -> List[Expression]:
        """Translate the terminals for a HighFunction BasicBlock."""
        bb = BasicBlock(basic_block)

        expressions = []

        for op in bb.pcodes():
            result, expression = self.translate_op(op)

            self.logger.debug(
                "op: '%s' -> result: %s expression: {%s}",
                op, result, expression
            )

            if result:
                expressions.append(expression)

        return expressions

    def translate_op(self, pcodeop) -> Tuple[bool, Optional[Expression | Branch]]:
        """Does the traslation of a specific Ghidra's PcodeOp into
        a terminal expression: it internally checks if the output
        is terminal (or if has not output) and then resolves the
        arguments as expression in order to build a statement."""
        output = pcodeop.getOutput()

        if output and not self.is_varnode_terminal(output):
            return False, None

        # the only exception is a unconditional branch
        # since that is encoded into the CFG
        if Opcode[pcodeop.getMnemonic()] == Opcode.BRANCH:
            return False, Branch(pcodeop.getInput(0))

        # FIXME: remove operation here or later?
        if Opcode[pcodeop.getMnemonic()] in [Opcode.INDIRECT, Opcode.MULTIEQUAL]:
            return False, None

        translation = self.do_translate_op(pcodeop)

        # here we have a possibly last expression
        # output = OPERATION(...)
        if output:
            return True, BinaryExpression(Operator.ASSIGNMENT, self.variable(output), translation)

        return True, translation

    def do_translate_op(self, pcodeop) -> UnaryExpression | BinaryExpression | Variable:
        """Recursively resolves the arguments"""
        self.logger.info("translating %s", pcodeop)
        args = []
        for idx_arg in range(pcodeop.getNumInputs()):
            arg = pcodeop.getInput(idx_arg)

            self.logger.info("resolving %s", arg)

            # check if it's a terminal, otherwise
            # try recursively to build an expression
            # by the defining PcodeOp
            if not self.is_varnode_terminal(arg):
                if arg.getDef() is None:
                    raise ValueError("Unexpected, we cannot find the terminal")
                arg = self.do_translate_op(arg.getDef())
            else:
                arg = self.variable(arg)

            args.append(arg)

        # now build the final expression with the pieces obtained
        return self.build_expression(Opcode[pcodeop.getMnemonic()], pcodeop, *args)

    def build_expression(self, opcode: Opcode, pcodeop, *args: Expression) -> Expression:
        match opcode, *args:
            # Control flow instructions
            case Opcode.CALL, FunctionRef(), *_:
                return FunctionCall(args[0], *args[1:])
            case Opcode.RETURN, value:  # TODO: it's possible to have more arguments?
                return UnaryExpression(Operator.RETURN, value)
            case Opcode.RETURN, Constant(), variable:
                return UnaryExpression(Operator.RETURN, variable)
            case Opcode.CBRANCH, Ram(), exp:
                return ControlFlowExpression(exp)

            # Boolean expressions
            case Opcode.INT_SLESS, left, right:
                return BinaryExpression(Operator.LESS_THAN, left, right)
            case Opcode.INT_EQUAL, left, right:
                return BinaryExpression(Operator.EQUAL, left, right)
            case Opcode.INT_NOTEQUAL, left, right:
                return BinaryExpression(Operator.NOT_EQUAL, left, right)

            # Arithmetics
            case Opcode.INT_ADD, left, right:
                return BinaryExpression(Operator.ADD, left, right)
            case Opcode.INT_MULT, left, right:
                return BinaryExpression(Operator.MULT, left, right)
            case Opcode.INT_SDIV, left, right:
                return BinaryExpression(Operator.DIV, left, right)

            # pseudo assignments
            case Opcode.CAST, variable:
                """A CAST performs identically to the COPY operator but also indicates that there is a forced change 
                in the data-types associated with the varnodes at this point in the code. The value input0 is 
                strictly copied into output; it is not a conversion cast. This operator is intended specifically for 
                when the value doesn't change but its interpretation as a data-type changes at this point. """
                # extract the data type of the output value
                return UnaryExpression(Operator.CAST, variable)
            case Opcode.COPY, variable:  # COPY is simpy renaming if it's not the terminal operation
                return variable
            case Opcode.INT_SEXT, variable:  # TODO: sign extension
                return variable
            case Opcode.INT_ZEXT, variable:  # TODO: zero extension
                return variable
            case Opcode.STORE, Parameter(), destination, source:
                # *destination = source
                return BinaryExpression(
                    Operator.ASSIGNMENT,
                    UnaryExpression(Operator.DEREF, destination),
                    source
                )
            case Opcode.LOAD, Parameter(), variable:
                # output = *variable
                return UnaryExpression(Operator.DEREF, variable)
            case Opcode.SUBPIECE, src, Constant() as count:
                return BinaryExpression(Operator.SUBPIECE, src, count)

            # Pointers
            case Opcode.PTRADD, base, offset, Constant() as size:
                """From the documentation: this operator serves as a more compact representation of the pointer 
                calculation, input0 + input1 * input2, but also indicates explicitly that input0 is a reference to an 
                array data-type. Input0 is a pointer to the beginning of the array, input1 is an index into the 
                array, and input2 is a constant indicating the size of an element in the array. As an operation, 
                PTRADD produces the pointer value of the element at the indicated index in the array and stores it in 
                output. """
                if not base.data_type.is_pointer:
                    raise ValueError("'%s' is not a pointer", base)

                if (points_to_size := base.data_type.points_to.size) != size.value:
                    self.logger.warning(
                        # raise ValueError(
                        "pointer arithmetic cannot be built: size={} != points_to.size={}".format(
                            size.value,
                            points_to_size,
                        )
                    )

                return BinaryExpression(Operator.ADD, base, offset)
            case Opcode.PTRSUB, base, Constant() as offset:
                """A PTRSUB performs the simple pointer calculation, input0 + input1, but also indicates explicitly 
                that input0 is a reference to a structured data-type and one of its subcomponents is being accessed. 
                Input0 is a pointer to the beginning of the structure, and input1 is a byte offset to the 
                subcomponent. As an operation, PTRSUB produces a pointer to the subcomponent and stores it in output."""
                # Note how in the documentation PTRSUB -> input0 + input1

                # here is tricky
                # in practice we have an operation that is building
                # a pointer of a data type from an address and an offset
                if isinstance(base, StackRef):  # TODO: wrt the doc seems that the stack pointer is treated differently
                    self.logger.warning("we have StackRef to resolve here!")
                    variable = self.resolve_variable_by_stack_offset(offset.value).getFirstStorageVarnode()
                    # here we use the PcodeOp's output data type to understand
                    # which component to use for the pointer of a composite type
                    data_type_output = pcodeop.getOutput().getHigh().getDataType()
                    # data_type_output_deref = data_type_output.getDataType()
                    return UnaryExpression(Operator.PTR, self.variable(variable))
                return BinaryExpression(Operator.FIELD_OF, base, offset)  # FIXME

            # put here the opcodes that are not handled
            case (Opcode.MULTIEQUAL | Opcode.INDIRECT), *_:
                self.logger.warning("%s for now is not handled", opcode)
                return NopExpression(Opcode.INDIRECT, *args)
            case _:
                raise ValueError(
                    "it's not possible to build expression for opcode: {} and args: {}",
                    opcode, args
                )

    def cfg(self) -> CodeBlock:
        """Build the CFG of the function"""

        # The procedure is kind of long since we want to create
        # a CFG where each node is a unique instance in order
        # to be able to create (at a later stage) structure code
        # in a more sane way
        class PlainNode(TypedDict):
            ins: List[int]
            outs: List[int]
            code: List[Expression]

        def _get_outs(_block):
            return [_block.getOut(_) for _ in range(_block.getOutSize())]

        def _get_ins(_block):
            return [_block.getIn(_) for _ in range(_block.getInSize())]

        def _get_start(_block) -> int:
            return _block.getStart().getOffset()

        head = self.context.getBasicBlocks()[0]
        address_start = _get_start(head)

        # first of all we need to collect all the info about nodes
        nodes: Dict[int, PlainNode] = {}

        for kind, block in traverse(head, childs=_get_outs):
            if kind != NodeType.NEW:
                continue

            node = {
                "ins" : list(map(_get_start, _get_ins(block))),
                "outs": list(map(_get_start, _get_outs(block))),
                'code': self.translate_block(block),
            }

            nodes[_get_start(block)] = node

        self.logger.debug(nodes)

        blocks: Dict[int, CodeBlock] = {}
        # now we can build the high-level CFG
        # first creating the nodes without in/out edges
        for start in nodes:
            node: PlainNode = nodes[start]
            blocks[start] = CodeBlock(start, node['code'])

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
        """Return the variable defined in the function."""
        return list(self._registry_named.values())
