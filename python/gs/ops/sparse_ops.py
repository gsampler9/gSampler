import torch

from ..format import _COO, _CSC, _CSR

target_mapping = {
    'u': 0,
    'e': 1,
    'v': 2,
    'src': 0,
    'edge': 1,
    'dst': 2
}


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
                shp1, shp2)
    # operands are padded to have the same dimensionality with leading 1's.
    if len(shp1) > len(shp2):
        pad_shp2 = (1,) * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = (1,) * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise "Feature shapes {} and {} are not valid for broadcasting.".format(
                shp1, shp2)
    rst = tuple(max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2))
    return rst[:-1] + (1,) if op == "dot" else rst


def _gspmm(gidx, op, reduce_op, u, e, u_target, on_format=_CSC):
    r"""Generalized Sparse Matrix Multiplication interface. It takes the result of
    :attr:`op` on source node feature and edge feature, leads to a message on edge.
    Then aggregates the message by :attr:`reduce_op` on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e)\in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``, ``copy_lhs``,
        ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``.
    u : tensor or None
        The feature on nodes, could be None if op is ``copy_rhs``.
    e : tensor or None
        The feature on edges, could be None if op is ``copy_lhs``.
    u_target : 0 or 1
        The target side of nodes, 0 denotes source nodes while 1 denotes destination nodes.

    Returns
    -------
    tuple
        The returned tuple is composed of two elements:
        - The first element refers to the result tensor.
        - The second element refers to a tuple composed of arg_u and arg_e
          (which is useful when reducer is `min`/`max`).

    Notes
    -----
    This function does not handle gradients.
    """
    use_u = op != "copy_rhs"
    use_e = op != "copy_lhs"
    if use_u and use_e:
        if u.device != e.device:
            raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                u.device, e.device)
        if u.dtype != e.dtype:
            raise "The node features' data type {} doesn't match edge features' data type {}, please convert them to the same type.".format(
                u.dtype, e.dtype)
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
    v_out_dim = gidx._CAPI_GetNumCols() if u_target == 0 else gidx._CAPI_GetNumRows()
    v_shp = (v_out_dim,) + infer_broadcast_shape(
        op, u_shp[1:], e_shp[1:]
    )
    v = torch.zeros(v_shp, dtype=dtype, device=device)
    use_cmp = reduce_op in ["max", "min"]
    arg_u, arg_e = None, None
    if use_cmp:
        if use_u:
            arg_u = torch.zeros(v_shp, dtype=torch.int64, device=device)
        if use_e:
            arg_e = torch.zeros(v_shp, dtype=torch.int64, device=device)
    if gidx._CAPI_GetNumEdges() > 0:
        gidx._CAPI_SpMM(op, reduce_op, u, e, v, arg_u, arg_e, u_target, on_format)
    # To deal with scalar node/edge features.
    if (expand_u or not use_u) and (expand_e or not use_e):
        v = torch.squeeze(v, -1)
    if expand_u and use_cmp:
        arg_u = torch.squeeze(arg_u, -1)
    if expand_e and use_cmp:
        arg_e = torch.squeeze(arg_e, -1)
    return v, (arg_u, arg_e)


def _gsddmm(gidx, op, lhs, rhs, lhs_target='u', rhs_target='v', on_format=_COO):
    r""" Generalized Sampled-Dense-Dense Matrix Multiplication interface. It
    takes the result of :attr:`op` on source node feature and destination node
    feature, leads to a feature on edge.

    .. math::
        x_{e} = \phi(x_u, x_e, x_v), \forall (u,e,v)\in \mathcal{G}

    where :math:`x_{e}` is the returned feature on edges and :math:`x_u`,
    :math:`x_v` refers to :attr:`u`, :attr:`v` respectively. :math:`\phi`
    is the binary operator :attr:`op`, and :math:`\mathcal{G}` is the graph
    we apply gsddmm on: :attr:`g`.

    Parameters
    ----------
    gidx : Backend C++ Graph
        The input graph index.
    op : str
        Binary operator, could be ``add``, ``sub``, ``mul``, ``div``, ``dot``.
    lhs : tensor or None
        Left hand operand.
    rhs : tensor or None
        Right hand operand.
    lhs_target : str
        The target of left hand operand, could be ``src``, ``edge``, ``dst``
        or their alias ``u``, ``e``, ``v``.
    rhs_target : str
        The target of right hand operand, could be ``src``, ``edge``, ``dst``
        or their alias ``u``, ``e``, ``v``.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function does not handle gradients.
    """
    use_lhs = op != "copy_rhs"
    use_rhs = op != "copy_lhs"
    if use_lhs and use_rhs:
        if lhs.device != rhs.device:
            raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                lhs.device, rhs.device)
        if lhs.dtype != rhs.dtype:
            raise "The operands data type don't match: {} and {}, please convert them to the same type.".format(
                lhs.dtype, rhs.dtype)
    # deal with scalar features.
    expand_lhs, expand_rhs = False, False
    if use_lhs and lhs.dim() == 1:
        lhs = torch.unsqueeze(lhs, -1)
        expand_lhs = True
    if use_rhs and rhs.dim() == 1:
        rhs = torch.unsqueeze(rhs, -1)
        expand_rhs = True
    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]

    device = lhs.device if use_lhs else rhs.device
    dtype = lhs.dtype if use_lhs else rhs.dtype
    lhs_shp = lhs.shape if use_lhs else (0,)
    rhs_shp = rhs.shape if use_rhs else (0,)
    out_shp = (gidx._CAPI_GetNumEdges(), ) +\
        infer_broadcast_shape(op, lhs_shp[1:], rhs_shp[1:])
    out = torch.zeros(out_shp, dtype=dtype, device=device)
    if gidx._CAPI_GetNumEdges() > 0:
        gidx._CAPI_SDDMM(op, lhs, rhs, out, lhs_target, rhs_target, on_format)
    if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
        out = torch.squeeze(out, -1)
    return out