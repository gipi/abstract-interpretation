import pytest
from abstract.analysis.graph import CFGNode, classify, graphivz, explore_all
from abstract.analysis.interpreter import CodeInterpreter
from abstract.pcode import Type, serialize_expression, ComponentType, deserialize_expressions
from .fixtures import cfg_loops, cfg_goto_you_said, func_encoding, func_smth
from abstract.analysis.code import StructureCode, CodeBlock, PlainProvider, \
    Function, BreakBlock, ContiguousBlock, WhileBlock


@pytest.mark.skip(reason="needs reworking")
def test_restructure_nested_loops():
    cfg = create_CFG(cfg_loops)

    assert cfg

    s = StructureCode(cfg)
    s.do_linearize()

    ccfg = s.get()

    assert ccfg.n_ins == 0
    assert ccfg.n_outs == 0

    code_interpreter = CodeInterpreter()

    code_interpreter.interpret(ccfg)

    print("\n".join(code_interpreter.get_lines()))


@pytest.mark.skip(reason="needs reworking")
def test_explore_goto():
    cfg = create_CFG(cfg_goto_you_said)

    assert cfg

    paths, nodes, loops = explore_all(cfg)

    for node in paths.keys():
        print(node)
        for path in paths[node]:
            print(" ", path)

    print(nodes)
    print(loops)

    s = StructureCode(cfg)

    s.do_linearize()
    s.detect_gotos()

    s.do_linearize()

    ccfg = s.get()

    code_interpreter = CodeInterpreter()

    code_interpreter.interpret(ccfg)


def test_encoding():
    plain_provider = PlainProvider(func_encoding)

    f = Function(plain_provider)

    s = StructureCode(f.cfg.get_head())

    s.do_linearize()

    ccfg = s.get()

    assert ccfg.n_outs == 0

    code = CodeInterpreter()

    code.interpret(ccfg)

    print("\n".join("{} {};".format(_.data_type.name, _.name) for _ in f.variables))
    print("\n".join(code.result()))


def test_smth():
    plain_provider = PlainProvider(func_smth)

    f = Function(plain_provider)

    s = StructureCode(f.cfg.get_head())

    s.do_linearize()

    head = s.get()

    assert head.n_outs == 0

    ccfg = s.get()

    code = CodeInterpreter()

    code.interpret(ccfg)

    print("\n".join("{} {};".format(_.data_type.name, _.name) for _ in f.variables))
    print("\n".join(code.result()))


def test_two_merging_blocks():
    # test only two blocks
    head = CodeBlock(0, None)
    tail = CodeBlock(1, None)

    head.ins = []
    head.outs = [tail]

    tail.ins = [head]
    tail.outs = []

    s = StructureCode(head)

    s.do_linearize()

    new_head = s.get()

    assert new_head.n_outs == 0
    assert new_head.n_ins == 0

    assert new_head.sl == head
    assert new_head.s == tail


def test_three_merging_blocks():
    # test only two blocks
    head = CodeBlock(0, None)
    middle = CodeBlock(1, None)
    tail = CodeBlock(2, None)

    head.ins = []
    head.outs = [middle]

    middle.ins = [head]
    middle.outs = [tail]

    tail.ins = [middle]
    tail.outs = []

    s = StructureCode(head)

    s.do_linearize()

    new_head = s.get()

    assert new_head.n_outs == 0
    assert new_head.n_ins == 0

    assert new_head.sl.__class__ == ContiguousBlock
    assert new_head.s == tail


def test_merging_blocks():
    # create a long line of blocks that will be merged
    head = prev = CodeBlock(0, None)
    prev.ins = []

    elements = [prev]

    COUNT = 10
    for idx in range(1, COUNT):
        block = CodeBlock(idx, None)

        elements.append(block)

        prev.outs = [block, ]

        block.ins = [prev]

        prev = block

    block.outs = []

    s = StructureCode(head)

    s.do_linearize()

    new_head = s.get()

    assert new_head
    assert new_head.__class__ == ContiguousBlock

    assert new_head.n_outs == 0

    head = new_head

    # now we have one less nesting of ContiguousBlocks
    for _ in range(COUNT - 1):
        assert head.__class__ == ContiguousBlock

        s = head.s

        assert s == elements[-(_ + 1)], "failed at iteration %d for elements %s" % (_, elements)
        assert s.__class__ == CodeBlock

        head = head.sl

    assert head.__class__ == CodeBlock


def test_while_block():
    entry_point = CodeBlock(0, None)
    head = CodeBlock(1, None)

    body = CodeBlock(2, None)

    exit = CodeBlock(3, None)

    entry_point.ins = []
    entry_point.outs = [head]

    head.ins = [entry_point, body]
    head.outs = [exit, body]

    body.ins = [head]
    body.outs = [head]

    exit.ins = [head]
    exit.outs = []

    # structure code starting from the head of the while
    s = StructureCode(head)

    it = s.iter_linearize()

    changed, new_head = next(it)

    assert changed
    assert new_head.n_ins == 1
    assert new_head.n_outs == 1


def test_while_block_bis():
    entry_point = CodeBlock(0, None)
    head = CodeBlock(1, None)

    body = CodeBlock(2, None)

    exit = CodeBlock(3, None)

    entry_point.ins = []
    entry_point.outs = [head]

    head.ins = [entry_point, body]
    head.outs = [exit, body]

    body.ins = [head]
    body.outs = [head]

    exit.ins = [head]
    exit.outs = []

    # structure code starting from the head of the while
    s = StructureCode(entry_point)

    it = s.iter_linearize()

    changed, new_head = next(it)

    assert changed
    assert new_head.n_ins == 0
    assert new_head.n_outs == 1

    while_block = new_head.outs[0]
    assert while_block.__class__ == WhileBlock
    assert while_block.n_ins == 1


def test_break_node():
    head = CodeBlock(0, None)
    first = CodeBlock(1, None)
    second = CodeBlock(2, None)

    head.ins = []
    head.outs = [first, second]

    first.ins = [head, ]
    first.outs = [second, ]

    second.ins = [head, first]
    second.outs = []

    # the creation of this block rework the graph structure
    bb = BreakBlock(first, second)

    assert head.n_ins == 0
    assert head.n_outs == 2

    assert first.n_ins == 1
    assert first.n_outs == 1

    assert first.outs == [bb, ]

    assert second.ins == [head, ]

    assert bb.n_ins == 1
    assert bb.ins == [first, ]
    assert bb.n_outs == 0
    assert bb.outs == []
    assert bb.breaks_to == second


def test_struct():
    # first of all create it
    int_type = Type('int', 4, [])
    struct_type = Type(
        'struct_miao',
        16,
        [
            ComponentType(0, 'a', int_type),
            ComponentType(4, 'b', int_type),
            ComponentType(8, 'c', int_type),
            ComponentType(12, 'd', int_type),
        ]
    )

    assert int_type != struct_type
    assert struct_type != int_type
    assert int_type != int

    assert struct_type['a'] == ComponentType(0, 'a', int_type)
    assert struct_type[0] == ComponentType(0, 'a', int_type)
    assert struct_type[1] == ComponentType(0, 'a', int_type)
    assert struct_type[2] == ComponentType(0, 'a', int_type)

    field0 = struct_type.get_component_at_offset(0)

    assert field0.name == 'a'

    field1 = struct_type.get_component_at_offset(1)

    assert field0 == field1

    # try to format
    assert "{0[6].name}".format(struct_type) == "b"

    serial_struct_type = serialize_expression(struct_type)

    # try to deserialize it
    assert deserialize_expressions(serial_struct_type) == struct_type
