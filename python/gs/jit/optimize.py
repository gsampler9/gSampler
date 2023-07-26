import operator
import torch.fx as fx
from typing import List, Tuple, Dict

node_kept = (
    "_CAPI_SDDMM",
    "_CAPI_SpMM",
    "_CAPI_FusedUOPV",
    "_CAPI_FusedESquareSum",
    "_CAPI_FusedEDivUSum",
)


def flatten(iter):
    ret = []
    for i in iter:
        if isinstance(i, List) or isinstance(i, Tuple):
            ret = ret + flatten(i)
        else:
            ret.append(i)
    return ret


def move_constant_to_top(gm: fx.GraphModule) -> fx.GraphModule:
    insert_place = None
    constant_nodes = []
    for i, node in enumerate(gm.graph.nodes):
        if node.op == "get_attr" and "constant" in node.name:
            constant_nodes.append(node)

        if i == 0:
            insert_place = node

    with gm.graph.inserting_before(insert_place):
        for n in constant_nodes:
            new_node = gm.graph.node_copy(n)
            n.replace_all_uses_with(new_node)

    gm.graph.lint()
    gm.recompile()
    return gm


def dce(gm: fx.GraphModule) -> fx.GraphModule:
    used_nodes_set = set()
    nodes_list = gm.graph.nodes
    for node in reversed(nodes_list):
        if node.op == "output" or node.op == "placeholder":
            used_nodes_set.add(node)

        if node in used_nodes_set or node.target in node_kept:
            for pre_node in flatten(node.args):
                if isinstance(pre_node, fx.Node):
                    used_nodes_set.add(pre_node)

                if isinstance(pre_node, Dict):
                    for _, value in pre_node.items():
                        if isinstance(value, fx.Node):
                            used_nodes_set.add(value)

                if isinstance(pre_node, List):
                    for value in pre_node:
                        if isinstance(value, fx.Node):
                            used_nodes_set.add(value)

                if isinstance(pre_node, Tuple):
                    for value in pre_node:
                        if isinstance(value, fx.Node):
                            used_nodes_set.add(value)

            for _, value in node.kwargs.items():
                if isinstance(value, fx.Node):
                    used_nodes_set.add(value)

                if isinstance(value, Dict):
                    for _, v in value.items():
                        if isinstance(v, fx.Node):
                            used_nodes_set.add(v)

                if isinstance(value, List):
                    for v in value:
                        if isinstance(v, fx.Node):
                            used_nodes_set.add(v)

                if isinstance(value, Tuple):
                    for v in value:
                        if isinstance(v, fx.Node):
                            used_nodes_set.add(v)

    for node in reversed(nodes_list):
        if node.target in node_kept:
            continue

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


def merge_relabel_and_all_indices(gm: fx.GraphModule) -> fx.GraphModule:
    gm = cse(gm)
    merge_dir = {}

    # scan
    for node in gm.graph.nodes:
        if node.target == "_CAPI_GetValidNodes" or node.target == "_CAPI_GraphRelabel":
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
            print(
                "{} has more than two children: {}, something wrong?".format(key, value)
            )

        with gm.graph.inserting_before(value[0]):
            new_relabel_node = None
            if value[0].target == "_CAPI_GraphRelabel":
                new_relabel_node = gm.graph.node_copy(value[0])
            else:
                new_relabel_node = gm.graph.node_copy(value[1])

            getitem_1 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 0)
            )
            getitem_2 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 1)
            )
            getitem_3 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 2)
            )
            getitem_4 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 3)
            )

            new_getitem_list = [getitem_1, getitem_2, getitem_3, getitem_4]

            for v in value:
                if v.target == "_CAPI_GetValidNodes":
                    # replace original all_idices
                    v.replace_all_uses_with(getitem_1)

                if v.target == "_CAPI_GraphRelabel":
                    # replace original gettitem of relabel
                    for i in v.users:
                        i.replace_all_uses_with(new_getitem_list[i.args[1]])

                    # replace original relabel
                    v.replace_all_uses_with(new_relabel_node)

    # remove dead code
    gm = dce(gm)
    return gm


def fuse_slicing_and_sampling(gm: fx.GraphModule) -> fx.GraphModule:
    # std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Slicing(seeds, axis, on_format, output_format)
    # std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Sampling(axis, fanout, replace, on_format, output_format)
    # std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> FusedSlicingSampling(axis, seeds, fanout, replace, on_format, output_format)

    def _get_graph_index_node(node):
        graph_node = None
        index_node = None

        candidates = list(node.users)

        for n in candidates:
            if n.args[1] == 0:
                graph_node = n
            else:
                index_node = n

        return graph_node, index_node

    for node in gm.graph.nodes:
        if (
            node.target == "_CAPI_Sampling"
            and node.args[0].args[0].target == "_CAPI_Slicing"
        ):
            slicing_node = node.args[0].args[0]
            slicing_graph_node, slicing_index_node = _get_graph_index_node(slicing_node)
            sampling_node = node
            sampling_graph_node, sampling_index_node = _get_graph_index_node(
                sampling_node
            )

            # check slicing and sampling for same axis
            if sampling_node.args[1] != slicing_node.args[2]:
                continue

            # check if graph_node are used by other nodes
            if len(slicing_graph_node.users) > 1:
                continue

            # check if index_noda are used by other nodes
            if slicing_index_node is not None:
                enable_fuse = True
                data_nodes = list(slicing_index_node.users)
                for n in data_nodes:
                    users = list(n.users)

                    if len(users) > 1:
                        enable_fuse = False
                        continue

                    if users[0].args[1] != sampling_index_node:
                        enable_fuse = False
                        continue

                if not enable_fuse:
                    continue

            # safe to fuse
            with gm.graph.inserting_before(slicing_node):
                fused_node = gm.graph.call_method(
                    "_CAPI_SlicingSampling",
                    args=(
                        slicing_node.args[0],
                        slicing_node.args[2],
                        slicing_node.args[1],
                        sampling_node.args[2],
                        sampling_node.args[3],
                        sampling_node.args[4],
                        sampling_node.args[5],
                    ),
                )

                fuse_graph_node = gm.graph.call_function(
                    operator.getitem, args=(fused_node, 0)
                )
                fuse_index_node = gm.graph.call_function(
                    operator.getitem, args=(fused_node, 1)
                )

                # replace graph
                sampling_graph_node.replace_all_uses_with(fuse_graph_node)

                # replace index (more complex)
                if slicing_index_node is not None:
                    new_data_nodes = []
                    be_repalced_nodes = []
                    for n in slicing_index_node.users:
                        data_node = gm.graph.call_function(
                            operator.__getitem__, args=(n.args[0], fuse_index_node)
                        )
                        new_data_nodes.append(data_node)

                        be_repalced_nodes.append(list(n.users)[0])

                    for new, old in zip(new_data_nodes, be_repalced_nodes):
                        old.replace_all_uses_with(new)

    # remove dead code
    gm = dce(gm)
    return gm


def fuse_ESqure_and_SumReduce(gm: fx.GraphModule) -> fx.GraphModule:
    def _get_spmm_node(tmp_node):
        for n in tmp_node.users:
            if n.target == operator.getitem and n.args[1] == 1:
                spmm_node = list(n.users)[0]
                break
        return spmm_node

    for node in gm.graph.nodes:
        if node.target == operator.pow and node.args[1] == 2:
            node_users = list(node.users)
            if len(node_users) == 2:
                if (
                    "_before_spmm" not in node_users[0].name
                    and "_before_spmm" not in node_users[1].name
                ):
                    continue

                if (
                    "_CAPI_SpMM" != node_users[0].target
                    and "_CAPI_SpMM" != node_users[1].target
                ):
                    continue

                ESqure_node = node
                tmp_node = node_users[0]
                spmm_node = _get_spmm_node(tmp_node)

                if spmm_node.args[1] != "copy_rhs" and spmm_node.args[2] != "sum":
                    continue

                # replace ESqure_node
                origin_node = ESqure_node.args[0]
                ESqure_node.replace_all_uses_with(origin_node)
                gm.graph.erase_node(ESqure_node)

                # replace spmm_node
                with gm.graph.inserting_before(spmm_node):
                    fused_e_squre_spmm_node = gm.graph.call_method(
                        "_CAPI_FusedESquareSum", args=(spmm_node.args)
                    )

                gm.graph.erase_node(spmm_node)

    return gm


def fuse_e_div_u_SumReduce(gm: fx.GraphModule) -> fx.GraphModule:
    # print(gm.graph)
    return gm


def merge_fused_u_mul_v(gm: fx.GraphModule) -> fx.GraphModule:
    merge_nodes = {}

    def _get_after_node(node):
        tmp_node = node.args[4]
        for after_node in tmp_node.users:
            if node != after_node:
                return after_node

        return None

    for node in gm.graph.nodes:
        if node.target == "_CAPI_SDDMM" and node.args[5] == 0 and node.args[6] == 2:
            key = str(node.args[0]) + str(node.args[1])

            if key in merge_nodes:
                merge_nodes[key].append(node)
            else:
                merge_nodes[key] = [node]

    for key, value in merge_nodes.items():
        if len(value) != 2:
            continue

        first_node = value[0]
        first_after_node = _get_after_node(first_node)
        second_node = value[1]

        with gm.graph.inserting_after(second_node):
            new_node = gm.graph.node_copy(first_after_node)
            first_after_node.replace_all_uses_with(new_node)
            gm.graph.erase_node(first_after_node)

        with gm.graph.inserting_before(second_node):
            fused_sddmm_u_op_v = gm.graph.call_method(
                "_CAPI_FusedUOPV",
                args=(
                    *first_node.args[0:5],
                    *second_node.args[2:5],
                    second_node.args[-1],
                ),
            )

            gm.graph.erase_node(first_node)
            gm.graph.erase_node(second_node)

    dce(gm)
    return gm
