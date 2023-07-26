import torch
from ..format import _COO, _CSC, _CSR

target_mapping = {"u": 0, "e": 1, "v": 2, "src": 0, "edge": 1, "dst": 2}


def infer_broadcast_shape(op, shp1, shp2):
    r"""Check the shape validity, and infer the output shape given input shape and operator.
    Note the both :attr:`shp1`, :attr:`shp2` and the returned shape are feature
    shapes (i.e. we remove the first dimension, which correspond to graph statistics
    such as number of nodes, number of edges, etc.).

    We allow applying op on operands with different shapes, according to the
    broadcasting semantics of Numpy/Scipy:
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    Parameters
    ----------
    op : str
        The binary op's name, could be `add`, `sub`, `mul`, `div`, `dot`, `copy_lhs`, `copy_rhs`.
    shp1 : tuple[int]
        The shape of lhs operand.
    shp2 : tuple[int]
        The shape of rhs operand.

    Returns
    -------
    tuple[int]
        shape after broadcasting
    """
    pad_shp1, pad_shp2 = shp1, shp2
    if op == "dot":
        if shp1[-1] != shp2[-1]:
            raise "Dot operator is only available for arrays with the same size on last dimension, but got {} and {}.".format(
                shp1, shp2
            )
    # operands are padded to have the same dimensionality with leading 1's.
    if len(shp1) > len(shp2):
        pad_shp2 = (1,) * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = (1,) * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise "Feature shapes {} and {} are not valid for broadcasting.".format(
                shp1, shp2
            )
    rst = tuple(max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2))
    return rst[:-1] + (1,) if op == "dot" else rst


def fused_u_mul_v(gidx, lhs1, rhs1, lhs2, rhs2, on_format=_COO):
    gidx = gidx._graph
    op = "mul"
    use_lhs = op != "copy_rhs"
    use_rhs = op != "copy_lhs"
    if use_lhs and use_rhs:
        if lhs1.device != rhs1.device:
            raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                lhs1.device, rhs1.device
            )
        if lhs1.dtype != rhs2.dtype:
            raise "The operands data type don't match: {} and {}, please convert them to the same type.".format(
                lhs1.dtype, rhs1.dtype
            )
    # deal with scalar features.
    expand_lhs, expand_rhs = False, False
    if use_lhs and lhs1.dim() == 1:
        lhs1 = torch.unsqueeze(lhs1, -1)
        lhs2 = torch.unsqueeze(lhs2, -1)
        expand_lhs = True
    if use_rhs and rhs1.dim() == 1:
        rhs1 = torch.unsqueeze(rhs1, -1)
        rhs2 = torch.unsqueeze(rhs2, -1)
        expand_rhs = True
    device = lhs1.device if use_lhs else rhs1.device
    dtype = lhs1.dtype if use_lhs else rhs1.dtype
    lhs_shp = lhs1.shape if use_lhs else (0,)
    rhs_shp = rhs1.shape if use_rhs else (0,)
    out_shp = (gidx._CAPI_GetNumEdges(),) + infer_broadcast_shape(
        op, lhs_shp[1:], rhs_shp[1:]
    )
    out1 = torch.zeros(out_shp, dtype=dtype, device=device)
    out2 = torch.zeros(out_shp, dtype=dtype, device=device)
    if gidx._CAPI_GetNumEdges() > 0:
        gidx._CAPI_FusedUOPV(op, lhs1, rhs1, out1, lhs2, rhs2, out2, on_format)
    if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
        out1 = torch.squeeze(out1, -1)
        out2 = torch.squeeze(out2, -1)
    return out1, out2


def e_square_sum(m, key, axis, on_format=_CSR) -> torch.Tensor:
    gidx = m._graph
    e = m.edata[key]
    if axis == 0:
        use_u = False
        use_e = True
        if use_u and use_e:
            if u.device != e.device:
                raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                    u.device, e.device
                )
            if u.dtype != e.dtype:
                raise "The node features' data type {} doesn't match edge features' data type {}, please convert them to the same type.".format(
                    u.dtype, e.dtype
                )
        # deal with scalar features.
        expand_u, expand_e = False, False
        if use_u and u.dim() == 1:
            u = torch.unsqueeze(u, -1)
            expand_u = True
        if use_e and e.dim() == 1:
            e = torch.unsqueeze(e, -1)
            expand_e = True
        device = u.device if use_u else e.device
        dtype = u.dtype if use_u else e.dtype
        u_shp = u.shape if use_u else (0,)
        e_shp = e.shape if use_e else (0,)
        v_out_dim = gidx._CAPI_GetNumCols()
        v_shp = (v_out_dim,)
        v = torch.zeros(v_shp, dtype=dtype, device=device)
        arg_u, arg_e = None, None
        if gidx._CAPI_GetNumEdges() > 0:
            gidx._CAPI_FusedESquareSum(
                "copy_rhs", "sum", None, e, v, None, None, 0, _CSC
            )
        # To deal with scalar node/edge features.
        v = torch.squeeze(v, -1)
        return v
    elif axis == 1:
        use_u = False
        use_e = True
        if use_u and use_e:
            if u.device != e.device:
                raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                    u.device, e.device
                )
            if u.dtype != e.dtype:
                raise "The node features' data type {} doesn't match edge features' data type {}, please convert them to the same type.".format(
                    u.dtype, e.dtype
                )
        # deal with scalar features.
        expand_u, expand_e = False, False
        if use_u and u.dim() == 1:
            u = torch.unsqueeze(u, -1)
            expand_u = True
        if use_e and e.dim() == 1:
            e = torch.unsqueeze(e, -1)
            expand_e = True
        device = u.device if use_u else e.device
        dtype = u.dtype if use_u else e.dtype
        u_shp = u.shape if use_u else (0,)
        e_shp = e.shape if use_e else (0,)
        v_out_dim = gidx._CAPI_GetNumRows()
        v_shp = (v_out_dim,)
        v = torch.zeros(v_shp, dtype=dtype, device=device)
        arg_u, arg_e = None, None
        if gidx._CAPI_GetNumEdges() > 0:
            gidx._CAPI_FusedESquareSum(
                "copy_rhs", "sum", None, e, v, None, None, 2, on_format
            )
        # To deal with scalar node/edge features.
        v = torch.squeeze(v, -1)
        return v
    else:
        raise "axis should be 0 or 1"


def e_div_u_sum(m, key, u, axis, on_format=_CSC) -> torch.Tensor:
    gidx = m._graph
    e = m.edata[key]
    if axis == 0:
        use_u = True
        use_e = True
        if use_u and use_e:
            if u.device != e.device:
                raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                    u.device, e.device
                )
            if u.dtype != e.dtype:
                raise "The node features' data type {} doesn't match edge features' data type {}, please convert them to the same type.".format(
                    u.dtype, e.dtype
                )
        # deal with scalar features.
        expand_u, expand_e = False, False
        if use_u and u.dim() == 1:
            u = torch.unsqueeze(u, -1)
            expand_u = True
        if use_e and e.dim() == 1:
            e = torch.unsqueeze(e, -1)
            expand_e = True
        device = u.device if use_u else e.device
        dtype = u.dtype if use_u else e.dtype
        u_shp = u.shape if use_u else (0,)
        e_shp = e.shape if use_e else (0,)
        v_out_dim = gidx._CAPI_GetNumCols()
        v_shp = (v_out_dim,)
        v = torch.zeros(v_shp, dtype=dtype, device=device)
        arg_u, arg_e = None, None
        if gidx._CAPI_GetNumEdges() > 0:
            gidx._CAPI_FusedEDivUSum("div", "sum", u, e, v, None, None, 0, on_format)
        # To deal with scalar node/edge features.
        v = torch.squeeze(v, -1)
        return v
    elif axis == 1:
        use_u = True
        use_e = True
        if use_u and use_e:
            if u.device != e.device:
                raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                    u.device, e.device
                )
            if u.dtype != e.dtype:
                raise "The node features' data type {} doesn't match edge features' data type {}, please convert them to the same type.".format(
                    u.dtype, e.dtype
                )
        # deal with scalar features.
        expand_u, expand_e = False, False
        if use_u and u.dim() == 1:
            u = torch.unsqueeze(u, -1)
            expand_u = True
        if use_e and e.dim() == 1:
            e = torch.unsqueeze(e, -1)
            expand_e = True
        device = u.device if use_u else e.device
        dtype = u.dtype if use_u else e.dtype
        u_shp = u.shape if use_u else (0,)
        e_shp = e.shape if use_e else (0,)
        v_out_dim = gidx._CAPI_GetNumRows()
        v_shp = (v_out_dim,)
        v = torch.zeros(v_shp, dtype=dtype, device=device)
        arg_u, arg_e = None, None
        if gidx._CAPI_GetNumEdges() > 0:
            # (const std::string& op, const std::string& reduce,
            #  torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
            #  torch::Tensor argu, torch::Tensor arge, int64_t u_target,
            #  int64_t on_format)
            gidx._CAPI_FusedEDivUSum("div", "sum", u, e, v, None, None, 2, on_format)
        # To deal with scalar node/edge features.
        v = torch.squeeze(v, -1)
        return v
    else:
        raise "axis should be 0 or 1"
