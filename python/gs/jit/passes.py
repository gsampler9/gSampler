import torch.fx as fx
from typing import List, Tuple


def flatten(iter):
    ret = []
    for i in iter:
        if isinstance(i, List) or isinstance(i, Tuple):
            ret = ret + flatten(i)
        else:
            ret.append(i)
    return ret


def dce(gm: fx.GraphModule) -> fx.GraphModule:
    used_nodes_set = set()
    nodes_list = gm.graph.nodes
    for node in reversed(nodes_list):
        if node.op == 'output' or node.op == 'placeholder':
            used_nodes_set.add(node)

        if node in used_nodes_set:
            for pre_node in flatten(node.args):
                if isinstance(pre_node, fx.Node):
                    used_nodes_set.add(pre_node)

            for _, value in node.kwargs.items():
                if isinstance(value, fx.Node):
                    used_nodes_set.add(value)

    for node in reversed(nodes_list):
        if node not in used_nodes_set:
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def cse(gm: fx.GraphModule) -> fx.GraphModule:
    nodes_list = gm.graph.nodes
    first_appear_ce_node = {}
    replace_nodes_set = set()
    for index, node in enumerate(nodes_list):
        key = str(node.target) + str(node.args) + str(node.kwargs)
        if key not in first_appear_ce_node:
            first_appear_ce_node[key] = node
        else:
            replace_nodes_set.add(node)

    for node in replace_nodes_set:
        key = str(node.target) + str(node.args) + str(node.kwargs)
        new_node = first_appear_ce_node[key]
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm