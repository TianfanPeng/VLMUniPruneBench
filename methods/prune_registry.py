PRUNE_FUNC = {}

def register_prune(name):
    def decorator(func):
        PRUNE_FUNC[name] = func
        return func
    return decorator
    

# 添加导入以触发注册
from . import Random_PreLLM
from . import GPrune_PreLLM
from . import DivPrune_PreLLM
from . import Pool_PreLLM

from . import Random_IntraLLM
from . import FastV_IntraLLM
from . import DART_IntraLLM
from . import FitPrune_IntraLLM
from . import Pdrop_IntraLLM