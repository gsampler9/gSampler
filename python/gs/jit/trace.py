import math
import torch
from torch.fx import Tracer
from torch.fx import GraphModule

from ..matrix_api import Matrix


class GSTracer(Tracer):

    def __init__(self,
                 autowrap_modules=(math, ),
                 autowrap_functions=(),
                 param_shapes_constant=False) -> None:
        super().__init__(autowrap_modules, autowrap_functions,
                         param_shapes_constant)

    def create_arg(self, a):
        if isinstance(a, Matrix):
            proxy = self.create_proxy('call_function', a.__class__,
                                      (a._graph, ), {}, None)
            return proxy.node
        else:
            return super().create_arg(a)


def gs_symbolic_trace(root, concrete_args=None) -> GraphModule:
    gs_tracer = GSTracer()
    graph = gs_tracer.trace(root, None)
    name = root.__class__.__name__ if isinstance(
        root, torch.nn.Module) else root.__name__
    gm = GraphModule(gs_tracer.root, graph, name)
    return gm