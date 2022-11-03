import logging
import sys

from . import getGlobalFunctions, toAddr
from .utils import decompile, CFG, Function


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger("abstract.utils").setLevel("INFO")
    function_name = sys.argv[1] if len(sys.argv) > 1 else 'main'

    print("Analyzing '%s'" % function_name)

    func = getGlobalFunctions(function_name)[0]

    res = decompile(func)

    # print(res.getDecompiledFunction().getC())
    # print(func.entryPoint)

    high_func = res.getHighFunction()

    f = Function(high_func)

    f.cfg.build(raw=True)

    # import ipdb;ipdb.set_trace()



