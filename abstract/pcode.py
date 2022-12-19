from __future__ import annotations

import itertools
import logging
from enum import Enum, auto
from typing import Optional, Protocol, TypeVar, List, Tuple, Union


logger = logging.getLogger(__name__)


T = TypeVar('T')


class Builder(Protocol):
    @classmethod
    def build(cls, from_ghidra):
        pass

    @classmethod
    def decode(cls, code: str):
        pass


class Operator(Enum):
    CALL = auto()
    RETURN = auto()
    ASSIGNMENT = auto()  # practically the 'var = EXPRESSION(...)'
    EQUAL = auto()  # logic equality
    NOT_EQUAL = auto()  # logic inequality
    LESS_THAN = auto()
    ADD = auto()
    SUB = auto()
    MULT = auto()
    DIV = auto()
    CAST = auto()  # change data type (think of it as a unary operator, like C)
    # handle pointers creation operations (like PTRADD and PTRSUB)
    # can be thought like the reference (&) operator in C (in some cases?)
    PTR  = auto()
    DEREF = auto()
    FIELD_OF = auto()
    ARRAY_OF = auto()

    #  TBD
    SUBPIECE = auto()


class UnaryExpression:
    MAP = {
        Operator.PTR: "&({argument})",
        Operator.DEREF: "*({argument})",
        Operator.RETURN: "return {argument}",
        Operator.CAST: "(<cast>){argument}",  # FIXME: this is binary
    }

    def __init__(self, operator: Operator, argument: Variable | UnaryExpression | BinaryExpression):
        self.operator = operator
        self.argument = argument

    def __repr__(self):
        return "<{}(operator={}, argument={})>".format(
            self.__class__.__name__,
            self.operator,
            str(self.argument),
        )

    def __str__(self):
        return self.MAP[self.operator].format(**vars(self))


class BinaryExpression:
    MAP = {
        Operator.EQUAL: "{left} == {right}",
        Operator.NOT_EQUAL: "{left} != {right}",
        Operator.LESS_THAN: "{left} < {right}",
        Operator.ADD: "{left} + {right}",
        Operator.SUB: "{left} - {right}",
        Operator.MULT: "{left} * {right}",
        Operator.DIV: "{left} / {right}",
        Operator.ASSIGNMENT: "{left} = {right}",
        # Operator.FIELD_OF: "&({left}->{left.data_type.points_to[right.value].name})",
        Operator.FIELD_OF: "&({left}->{right})",
        Operator.ARRAY_OF: "{left}[{right}]",
        Operator.SUBPIECE: "Subpiece of {left} from {right}"
    }

    def __init__(self, operator: Operator, left: Expression, right: Expression):
        self.operator = operator
        self.left = left
        self.right = right

        #if self.operator == Operator.FIELD_OF:
        #    self.field_name = self.left.data_type.points_to.get_field_at_offset(self.right.value)[1]

    def __repr__(self):
        return "<{}(operator={}, left={}, right={})>".format(
            self.__class__.__name__,
            self.operator,
            str(self.left),
            str(self.right),
        )

    def __str__(self):
        if self.operator == Operator.FIELD_OF:
            breakpoint()
        return self.MAP[self.operator].format(**vars(self))

    @property
    def data_type(self) -> Type:
        match self.operator, self.left, self.right:
            case Operator.FIELD_OF, base, Constant(value=offset):
                """This is originated by PTRxxx, so the resulting type should be
                a pointer to the data type of the corresponding field."""
                ptr_struct_type = base.data_type

                if not ptr_struct_type.is_pointer:
                    raise ValueError("I was expecting a ptr type for the left argument of '%s', instead I have '%s'" % (
                        self, ptr_struct_type))

                struct_type = ptr_struct_type.points_to

                if not struct_type.is_composite:
                    raise ValueError("I was expecting a composite type for the left argument of '%s'" % self)

                breakpoint()

                component = struct_type.get_component_at_offset(offset)

                if offset != component.offset:
                    raise ValueError("no subfield found at offset {} for data type '{}'".format(offset, struct_type))

                return Type("{} *".format(component.data_type.name),
                            8,  # FIXME: get a generic way of defining size of pointers
                            [], points_to=component.data_type)
            case _:
                return self.left.data_type  # FIXME


class ControlFlowExpression:
    """Wrap a control flow expression"""
    def __init__(self, expression: Expression):
        self.expression = expression

    def __str__(self):
        return str(self.expression)

    def __repr__(self):
        return "<{}(expression={})>".format(self.__class__, self.expression)


class FunctionCall:
    def __init__(self, func: FunctionRef, *args: Expression):
        self.ref = func
        self.args = args

    def __repr__(self):
        return "<{}(ref={}, args={})".format(
            self.__class__,
            self.ref,
            ",".join([str(_) for _ in self.args]),
        )

    def __str__(self):
        return "{}({})".format(
            self.ref.name,
            ",".join(str(_) for _ in self.args),
        )


class Variable:
    """Wrap the varnode data type in order to be able
    to be manageable."""
    def __init__(self, name, data_type: Type):
        self.name = name
        self.data_type = data_type
        self.size = self.data_type.size

    def __repr__(self):
        return "<{}(name={},data_type={})>".format(
            self.__class__.__name__,
            self.name,
            self.data_type,
        )

    def __str__(self):
        return self.name

    def __members(self):
        return self.space, self.offset, self.size

    def __eq__(self, other):
        if type(self) == type(other):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())

    @classmethod
    def decode(cls, code: str) -> Variable:
        return cls(None, None, None, None, None, code)

    @classmethod
    def build(cls, varnode) -> Variable:
        # there is getSpace() but it's spaceID that
        # I don't know what is
        space = varnode.getAddress().getAddressSpace().getType()
        offset = varnode.getOffset()
        size = varnode.getSize()

        high = varnode.getHigh()

        data_type = high.getDataType() if high else None
        var_type: VariableType = VariableType.map(high) if high else None
        is_named = high and (n := high.getName()) and n != "UNNAMED"
        name = n if is_named else "{}_{:x}".format(cls.address_space(space), offset)

        return cls(space, offset, size, var_type, data_type, name)


class FieldVariable:
    def __init__(self, parent: Variable, name: str):
        self.parent = parent
        self.name = name
        self.data_type = parent.data_type.get_field(name)
        self.size = self.data_type.size

    def __str__(self):
        return "{}.{}".format(
            self.parent.name,
            self.name,
        )

    def __repr__(self):
        return "<{}(parent={}, name={})>".format(
            self.__class__,
            self.parent.name,
            self.name,
        )


class TypeIsNotCompositeError(TypeError):
    pass


class TypeHasNotField(TypeError):
    pass


class Type:
    """Describe a type"""

    def __init__(self, name: str, size: int, components: List[ComponentType], points_to: Optional[Type] = None):
        self.name = name
        self.size = size
        self.components = components
        self.points_to = points_to

    def __repr__(self):
        return "<{}(name={},size={},components={},points_to={})>".format(
            self.__class__.__name__,
            self.name,
            self.size,
            [str(_) for _ in self.components],
            str(self.points_to),
        )

    def __str__(self):
        return self.name

    def __getitem__(self, item) -> ComponentType:
        if not self.is_composite:
            raise TypeIsNotCompositeError("Type subclass '{}' is not composite".format(self.__class__))

        if type(item) == str:
            if item not in [_.name for _ in self.components]:
                raise TypeHasNotField("field '{}' not found for Type '{}'".format(item, self.__class__))

            return self.get_component_by_name(item)

        if type(item) == int:
            if item > self.size:
                raise TypeHasNotField("field at offset'{}' not found for Type '{}'".format(item, self.__class__))

            return self.get_component_at_offset(item)

        raise TypeHasNotField("the key '{}' must be a field name or an offset".format(item))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               self.name == other.name and self.size == other.size and self.components == other.components

    @property
    def is_composite(self):
        return bool(self.components)

    def get_component_at_offset(self, offset: int) -> ComponentType:
        """return the nearest field that contain that offset, it's the
        caller responsability to check it matches"""
        if not self.is_composite:
            raise TypeIsNotCompositeError("Type subclass '{}' is not composite".format(self.__class__))

        if offset >= self.size:
            raise ValueError("offset requested ({}) is over the size of Type '{}'".format(offset, self.name))

        for start, end in itertools.pairwise(self.components):
            if start.offset <= offset < end.offset:
                return start
        else:
            return end  # we checked that offset < size so the last component is our choice

    def get_component_by_name(self, name: str) -> ComponentType:
        if not self.is_composite:
            raise TypeError("'{}' is not a composite type".format(self.name))

        for component in self.components:
            if component.name == name:
                return component
        else:
            raise ValueError("no field with name '{}'".format(name))

    def get_component_data_type_by_name(self, name: str):
        return self.get_component_by_name(name).data_type

    @property
    def is_pointer(self) -> bool:
        return bool(self.points_to)


Void = Type('void', 4, [])  # FIXME: size


class ComponentType:

    def __init__(self, offset: int, name: str, data_type: Type):
        self.offset = offset
        self.name = name
        self.data_type = data_type

    def __repr__(self):
        return "<{}(offset=0x{:x},name={},data_type={})>".format(
            self.__class__.__name__,
            self.offset,
            self.name,
            str(self.data_type),
        )

    def __str__(self):
        return "({}){}@{}".format(self.data_type, self.name, self.offset)

    def __eq__(self, other):
        return self.name == other.name and self.offset == other.offset and self.data_type == other.data_type


class Constant:
    def __init__(self, value: int, data_type: Type):
        self.value = value
        self.data_type = data_type

    def __repr__(self):
        return "<{}(value=0x{:x}, data_type={})>".format(
            self.__class__.__name__,
            self.value,
            str(self.data_type),
        )

    def __str__(self):
        if self.data_type.name != 'int':
            return "({}){:x}".format(self.data_type.name, self.value)

        return "0x{:x}".format(self.value)


class FunctionRef:
    def __init__(self, name: str, *signature: Type, return_type: Type = Void):
        self.name = name
        self.signature = signature
        self.return_type = return_type

    def __str__(self):
        return "{}({})".format(
            self.name,
            ",".join(str(_) for _ in self.signature),
        )


class NopExpression:
    """Expression without effect"""
    def __init__(self, opcode, *args):
        self.opcode = opcode
        self.args = args

    def __repr__(self):
        return "<{}(opcode={}, args={})>".format(
            self.__class__.__name__,
            self.opcode,
            [str(_) for _ in self.args],
        )


Expression = Union[
    Variable,
    FieldVariable,
    UnaryExpression,
    BinaryExpression,
    ControlFlowExpression,
    Constant,
    FunctionCall,
    NopExpression,
]


def serialize_expression(expression: Expression | Type) -> Tuple:
    """Recursively resolves a AST-like for expressions"""
    logger.debug("serialize(%s)", repr(expression))
    match expression:
        case None:
            return 'None'
        case Constant(value=value, data_type=data_type):
            return 'Constant', value, serialize_expression(data_type)
        case Variable(name=name, data_type=data_type):
            return 'Variable', name, serialize_expression(data_type)
        case Type(name=name, size=size, components=components, points_to=points_to):
            return 'Type', name, size, [serialize_expression(_) for _ in components], serialize_expression(points_to)
        case ComponentType(offset=offset, name=name, data_type=data_type):
            return 'ComponentType', offset, name, serialize_expression(data_type)
        case UnaryExpression(operator=op, argument=arg):
            return op.name, serialize_expression(arg)
        case BinaryExpression(operator=op, left=left, right=right):
            return op.name, serialize_expression(left), serialize_expression(right)
        case ControlFlowExpression(expression=expr):
            return 'ControlFlowExpression', serialize_expression(expr)
        case _:
            raise ValueError("unable to serialize {}".format(repr(expression)))


def deserialize_expressions(elements: Tuple) -> None | Expression | Type:
    if elements is None or elements == 'None':
        return None

    identifier, *args = elements
    match identifier, *args:
        case 'Constant', value, data_type:
            return Constant(value, deserialize_expressions(data_type))
        case 'Variable', name, data_type:
            return Variable(name, deserialize_expressions(data_type))
        case 'Type', name, size, list() as components, points_to:
            return Type(name, size, [deserialize_expressions(_) for _ in components], points_to=deserialize_expressions(points_to))
        case 'ComponentType', offset, name, data_type:
            return ComponentType(offset, name, deserialize_expressions(data_type))
        case 'ControlFlowExpression', expression:
            return ControlFlowExpression(deserialize_expressions(expression))
        case operator_name, argument:
            return UnaryExpression(Operator[operator_name], deserialize_expressions(argument))
        case operator_name, left, right:
            return BinaryExpression(Operator[operator_name], deserialize_expressions(left), deserialize_expressions(right))
        case _:
            raise ValueError("unable to deserialize {}".format(elements))
