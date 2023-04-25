import torch
from gs.format import _COO, _CSC, _CSR, _DCSC, _DCSR

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
    if lhs.device != rhs.device:
        raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
            lhs.device, rhs.device)
    if lhs.dtype != rhs.dtype:
        raise "The operands data type don't match: {} and {}, please convert them to the same type.".format(
            lhs.dtype, rhs.dtype)
    # deal with scalar features.
    expand_lhs, expand_rhs = False, False
    if lhs.dim() == 1:
        lhs = torch.unsqueeze(lhs, -1)
        expand_lhs = True
    if rhs.dim() == 1:
        rhs = torch.unsqueeze(rhs, -1)
        expand_rhs = True
    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]

    out_shp = (gidx._CAPI_get_num_edges(), ) +\
        infer_broadcast_shape(op, lhs.shape[1:], rhs.shape[1:])
    out = torch.zeros(out_shp, dtype=lhs.dtype, device=lhs.device)
    if gidx._CAPI_get_num_edges() > 0:
        gidx._CAPI_sddmm(op, lhs, rhs, out, lhs_target, rhs_target, on_format)
    if expand_lhs and expand_rhs:
        out = torch.squeeze(out, -1)
    return out


def gsddmm_internal(gidx, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v', on_format=_COO):
    if op == 'sub':
        op = 'add'
        rhs_data = -rhs_data
    if op == 'div':
        op = 'mul'
        rhs_data = 1. / rhs_data
    return _gsddmm(gidx, op, lhs_data, rhs_data, lhs_target, rhs_target, on_format)
