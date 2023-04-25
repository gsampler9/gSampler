import operator
import tarfile
from .passes import dce
import torch.fx as fx


def merge_relabel_and_all_indices(gm: fx.GraphModule) -> fx.GraphModule:
    merge_dir = {}

    # scan
    for node in gm.graph.nodes:
        if node.target == '_CAPI_all_indices' or node.target == '_CAPI_relabel':
            # node.args[0] is the parent of node
            if node.args[0] not in merge_dir:
                merge_dir[node.args[0]] = [node]
            else:
                merge_dir[node.args[0]].append(node)

    # begin merge
    for key, value in merge_dir.items():

        if len(value) < 2:
            continue

        if len(value) > 2:
            print("{} has more than two children: {}, something wrong?".format(
                key, value))

        with gm.graph.inserting_before(value[0]):
            new_relabel_node = None
            if value[0].target == '_CAPI_relabel':
                new_relabel_node = gm.graph.node_copy(value[0])
            else:
                new_relabel_node = gm.graph.node_copy(value[1])

            getitem_1 = gm.graph.call_function(operator.getitem,
                                               args=(new_relabel_node, 0))
            getitem_2 = gm.graph.call_function(operator.getitem,
                                               args=(new_relabel_node, 1))
            getitem_3 = gm.graph.call_function(operator.getitem,
                                               args=(new_relabel_node, 2))

            new_getitem_list = [getitem_1, getitem_2, getitem_3]

            for v in value:
                if v.target == '_CAPI_all_indices':
                    # replace original all_idices
                    v.replace_all_uses_with(getitem_1)

                if v.target == '_CAPI_relabel':
                    # replace original gettitem of relabel
                    for i in v.users:
                        i.replace_all_uses_with(new_getitem_list[i.args[1]])

                    # replace original relabel
                    v.replace_all_uses_with(new_relabel_node)

    # remove dead code
    gm = dce(gm)
    return gm


def fuse_slicing_and_sampling(gm):
    """
    Fuses columnwise_slicing and columnwise_sampling
    """
    for node in gm.graph.nodes:
        if node.target == '_CAPI_columnwise_sampling' and node.args[
                0].target == '_CAPI_columnwise_slicing':
            if len(node.args[0].users) > 1:
                continue
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_method(
                    '_CAPI_fused_columnwise_slicing_sampling',
                    args=(
                        *node.args[0].args,
                        *node.args[1:],
                    ))
                node.replace_all_uses_with(new_node)

    # remove dead code
    gm = dce(gm)
    return gm