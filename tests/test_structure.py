from abstract.analysis.graph import CFGNode, classify
from .fixtures import cfg_loops
from abstract.analysis.code import StructureCode, CodeBlock, PlainNode, PlainCFG


def create_CFG(cfg: PlainCFG) -> CFGNode:
    CFG = {}
    # first we create all the nodes
    for start in cfg:
        CFG[start] = CodeBlock(start)

    # then we setup the ins/outs accordingly
    for start in cfg:
        node = CFG[start]
        ins = cfg[start]['ins']
        node.ins = list(map(
            lambda _: CFG[_],
            ins,
        ))
        outs = cfg_loops[start]['outs']
        node.outs = list(map(
            lambda _: CFG[_],
            outs,
        ))

    return CFG[list(CFG.keys())[0]]


def test_restructure():
    cfg = create_CFG(cfg_loops)

    assert cfg

    forest, edges, loops = classify(cfg)

    s = StructureCode(cfg, loops)

    print("\n".join(s.get().code()))
