import lark.exceptions
import pytest

from abstract.analysis.ast import get_parser


def test_ambiguity():
    program = """
if (x < 1)
    if (x < 2)
        x = x + 10;
else
    x = 0;
"""
    parser = get_parser()

    tree = parser.parse(program)

    assert tree

    print(tree)
    print(tree.pretty())


def test_need_semicolon():
    parser = get_parser()

    with pytest.raises(lark.exceptions.UnexpectedToken) as e:
        tree = parser.parse("break")

        print(tree.pretty())

    assert "Expected one of:" in str(e.value)
    assert "* SEMICOLON" in str(e.value)


def test_isolated_aexpr():
    parser = get_parser()

    with pytest.raises(lark.exceptions.UnexpectedToken) as e:
        tree = parser.parse("""x + 1;""")

        print(tree.pretty())

    assert "Expected one of:" in str(e.value)
    assert "* ASSIGN" in str(e.value)


def test_keywords():
    """Shouldn't be possible to use keywords as variables names."""
    parser = get_parser()

    for keyword in ["break", "while", "if"]:
        with pytest.raises(lark.exceptions.UnexpectedToken) as e:
            tree = parser.parse(f"{keyword} = 1;")

            print(tree.pretty())

        assert "Unexpected token" in str(e.value)
