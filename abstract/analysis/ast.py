"""
AST-like handling module
"""
import dataclasses
from enum import Enum, auto
from pprint import pprint
from typing import Optional, Union

from lark import Lark, Transformer, v_args, Discard, Token, Tree


# partly inspired from pg.51 of "Principles of Abstract interpretation"
GRAMMAR = """
SEMICOLON: ";"
LPAREN : "("
RPAREN : ")"

LBRACE : "{"
RBRACE : "}"

MINUS : "-"
PLUS : "+"
ASSIGN : "="
LT : "<"
NAND : "!&"


program : statementlist

statementlist: statement+

statement : if_statement
          | while_statement
          | break_statement
          | block_statement
          | var ASSIGN aexpr SEMICOLON
          | SEMICOLON

block_statement : LBRACE statementlist RBRACE
if_statement : IF LPAREN bexpr RPAREN statement
             | IF LPAREN bexpr RPAREN statement ELSE statement
while_statement : WHILE LPAREN bexpr RPAREN statement
break_statement: BREAK SEMICOLON

?aexpr : value
       | var
       | aexpr MINUS aexpr
       | MINUS aexpr
       | LPAREN aexpr RPAREN
       | aexpr PLUS aexpr      -> sum

bexpr : aexpr LT aexpr
      | bexpr NAND bexpr
      | LPAREN bexpr RPAREN

var   : WORD
value : SIGNED_INT
BREAK : "break"
WHILE : "while"
IF    : "if"
ELSE  : "else"


%import common.SIGNED_INT
%import common.CNAME
%import common.WORD
%import common.WS
%ignore WS
"""


class Variable:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"


# TODO: catch syntax error, like `break` outside a `while`
class ASTFirstStageTransformer(Transformer):
    @v_args(inline=True)
    def var(self, word):
        # `word` is a Token, subclass of str
        return Variable(word.value)

    @v_args(inline=True)
    def value(self, integer):
        return int(integer.value)

    @v_args(inline=True)
    def sum(self, left, op, right):
        # we want something like `left - ( - right)`
        # so we are going to manually build the corresponding tree
        inner = Tree(
            data='aexpr',
            children=[
                Token('MINUS', '-'),
                right,
            ],
        )

        return Tree(
            data='aexpr',
            children=[
                left,
                Token('MINUS', '-'),
                inner,
            ],
        )

    @v_args(inline=True)
    def block_statement(self, statementlist):
        """Reduce a block statement to its inner statementlist"""
        return statementlist

    def LBRACE(self, *args):
        return Discard

    def RBRACE(self, *args):
        return Discard

    def LPAREN(self, *args):
        return Discard

    def RPAREN(self, *args):
        return Discard

    def SEMICOLON(self, *args):
        return Discard


@dataclasses.dataclass
class AExpr:
    left: Union['AExpr', int]
    right: Optional[Union['AExpr', int]] = None

    @property
    def is_unary(self):
        return self.right is None

    @staticmethod
    def parenthesize(param):
        """Check if it's composite and parenthesize it in case."""
        return f"{param}" if isinstance(param, (int, Variable)) else f"({param})"

    def __repr__(self):
        return f"<{self.__class__.__name__}(left={self.left}, right={self.right})"

    def __str__(self):
        # the tricky thing is to understand where to place parenthesis
        # but remember, you don't know where you are placed in the tree
        # but you know that the internal aexpr are below you, so makes
        # sense for you to put parenthesis to your children
        left = self.parenthesize(self.left)

        if self.is_unary:
            return f"-{left}"

        right = self.parenthesize(self.right)

        return f"{left} - {right}"


class BOp(Enum):
    NAND = auto()
    LT   = auto()


@dataclasses.dataclass
class BExpr:
    left: 'BExpr'
    op: Optional[BOp] = None
    right: Optional['BExpr'] = None

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

@dataclasses.dataclass
class Statement:
    label: Optional[str]


@dataclasses.dataclass
class IfStatement(Statement):
    expr: BExpr
    block: Statement

    def __str__(self):
        return f"if ({self.expr}) {self.block}"


@dataclasses.dataclass
class BreakStatement(Statement):
    pass


class ASTFinalStageTransformer(Transformer):
    COUNTER = 0

    @v_args(inline=True)
    def aexpr(self, left, op, right=None):
        match left:
            case Token(type='MINUS', value='-'):
                return AExpr(left=op)
            case _:
                return AExpr(left=left, right=right)

    @v_args(inline=True)
    def bexpr(self, left, op, right=None):
        match left:
            case Token(type='NAND', value='!&'):
                return BExpr(left=op)
            case _:
                op = BOp[op.type]
                return BExpr(left=left, op=op, right=right)

    @v_args(inline=True)
    def if_statement(self, if_token, bexpr, statement, else_token=None, else_statement=None):
        if else_token and else_statement:
            raise NotImplemented

        return IfStatement(label='1', expr=bexpr, block=statement)

    @v_args(inline=True)
    def break_statement(self, value):
        return BreakStatement(label='2')


def get_parser():
    return Lark(GRAMMAR, parser='lalr', start='program', debug=True)


if __name__ == "__main__":
    parser = Lark(GRAMMAR, start='program', debug=True)

    program = """x = 1;
while (x < 10) {
    x = x + 1;
    ;
    if (x < 5)
        break;
}
x = x - 2;
"""

    parsed = parser.parse(program)
    print(parsed.pretty())
    print("# normal")
    pprint(parsed)

    print("# transformed")
    first_stage = ASTFirstStageTransformer().transform(parsed)
    print(first_stage)
    print(first_stage.pretty())

    final_stage = ASTFinalStageTransformer().transform(first_stage)

    print(final_stage.pretty())