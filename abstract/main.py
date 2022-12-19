import logging
import sys
from pprint import pprint

import ipdb

from abstract.analysis.interpreter import CodeInterpreter
from abstract.utils import traverse, NodeType
from .analysis.ghidra import getGlobalFunctions, toAddr, currentLocation
from .analysis.code import Function, StructureCode
from .analysis.ghidra import decompile, GhidraProvider
from .analysis.graph import graphivz

if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger("abstract").setLevel("DEBUG")
    cmd = sys.argv[1]
    function_name = sys.argv[2] if len(sys.argv) > 1 else 'main'

    print("Analyzing '%s'" % function_name)

    ghidra_func = getGlobalFunctions(function_name)[0]

    res = decompile(ghidra_func)
    high_function = res.getHighFunction()
    gp = GhidraProvider(high_function)

    f = Function(gp)

    if cmd == "graph":
        graphivz(high_function.getFunction().getName(), f.cfg.get_head())
    elif cmd == "json":
        pprint(f.json())
    elif cmd == 'lift':
        cfg = f.cfg

        s = StructureCode(cfg.get_head())

        s.do_linearize()

        ccfg = s.get()

        code = CodeInterpreter()

        code.interpret(ccfg)

        print("\n".join("{} {};".format(_.data_type.name, _.name) for _ in f.variables))
        print("\n".join(code.result()))

    ipdb.set_trace()



