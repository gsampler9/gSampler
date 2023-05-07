import torch
import os

package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
so_path = os.path.join(package_path, 'libgs.so')
torch.classes.load_library(so_path)

from .matrix_api import Matrix
from .jit import GSTracer, gs_symbolic_trace, compile
from . import ops
from .format import _COO, _CSC, _CSR

Graph = torch.classes.gs_classes.Graph
#HeteroGraph = torch.classes.gs_classes.HeteroGraph
