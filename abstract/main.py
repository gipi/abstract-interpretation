import logging
import sys

from . import getGlobalFunctions, toAddr
from .analysis.code import decompile, Function


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger("abstract.utils").setLevel("INFO")
    function_name = sys.argv[1] if len(sys.argv) > 1 else 'main'
    cmd = sys.argv[1]
    function_name = sys.argv[2] if len(sys.argv) > 1 else 'main'

    print("Analyzing '%s'" % function_name)

    func = getGlobalFunctions(function_name)[0]

    res = decompile(func)

    # print(res.getDecompiledFunction().getC())
    # print(func.entryPoint)

    high_func = res.getHighFunction()

    f = Function(high_func)

    if cmd == "graph":
        f.graphivz()
    elif cmd == "display":
        f.cfg.ops(raw=False)
    elif cmd == "loops":
        f.cfg.loops()
    elif cmd == "ifs":
        f.cfg.linearize()
    elif cmd == "json":
        print(f.cfg.json())

    # import ipdb;ipdb.set_trace()



