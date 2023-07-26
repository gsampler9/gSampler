from itertools import product
import sys
import torch

from ..format import _COO, _CSC, _CSR

torch.fx.wrap("_before_sddmm")
torch.fx.wrap("_after_sddmm")

__all__ = ["gsddmm"]


target_mapping = {"u": 0, "e": 1, "v": 2, "src": 0, "edge": 1, "dst": 2}


def _before_sddmm(num_edges, op, lhs_data, rhs_data):
    if op not in ["copy_lhs", "copy_rhs"]:
        lhs_data, rhs_data = reshape_lhs_rhs(lhs_data, rhs_data)

    if op == "sub":
        op = "add"
        rhs_data = -rhs_data
    if op == "div":
        op = "mul"
        rhs_data = 1.0 / rhs_data

    lhs = lhs_data
    rhs = rhs_data

    use_lhs = op != "copy_rhs"
    use_rhs = op != "copy_lhs"
    if use_lhs and use_rhs:
        if lhs.device != rhs.device:
            raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                lhs.device, rhs.device
            )
        if lhs.dtype != rhs.dtype:
            raise "The operands data type don't match: {} and {}, please convert them to the same type.".format(
                lhs.dtype, rhs.dtype
            )

    # deal with scalar features.
    expand_lhs, expand_rhs = False, False
    if use_lhs and lhs.dim() == 1:
        lhs = torch.unsqueeze(lhs, -1)
        expand_lhs = True
    if use_rhs and rhs.dim() == 1:
        rhs = torch.unsqueeze(rhs, -1)
        expand_rhs = True

    device = lhs.device if use_lhs else rhs.device
    dtype = lhs.dtype if use_lhs else rhs.dtype
    lhs_shp = lhs.shape if use_lhs else (0,)
    rhs_shp = rhs.shape if use_rhs else (0,)
    out_shp = (num_edges,) + infer_broadcast_shape(op, lhs_shp[1:], rhs_shp[1:])
    out = torch.zeros(out_shp, dtype=dtype, device=device)
    condition = (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs)

    return (lhs, rhs, out, condition)


def _after_sddmm(out, condition):
    if condition:
        out = torch.squeeze(out, -1)
    return out


def infer_broadcast_shape(op, shp1, shp2):
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


def reshape_lhs_rhs(lhs_data, rhs_data):
    lhs_shape = lhs_data.shape
    rhs_shape = rhs_data.shape
    if len(lhs_shape) != len(rhs_shape):
        max_ndims = max(len(lhs_shape), len(rhs_shape))
        lhs_pad_ndims = max_ndims - len(lhs_shape)
        rhs_pad_ndims = max_ndims - len(rhs_shape)
        new_lhs_shape = (lhs_shape[0],) + (1,) * lhs_pad_ndims + lhs_shape[1:]
        new_rhs_shape = (rhs_shape[0],) + (1,) * rhs_pad_ndims + rhs_shape[1:]
        lhs_data = lhs_data.view(new_lhs_shape)
        rhs_data = rhs_data.view(new_rhs_shape)
    return lhs_data, rhs_data


def gsddmm(g, op, lhs_data, rhs_data, lhs_target="u", rhs_target="v", on_format=_COO):
    num_edges = g._graph._CAPI_GetNumEdges()
    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]
    (lhs, rhs, out, condition) = _before_sddmm(num_edges, op, lhs_data, rhs_data)
    g._graph._CAPI_SDDMM(op, lhs, rhs, out, lhs_target, rhs_target, on_format)
    out = _after_sddmm(out, condition)
    return out


def _gen_sddmm_func(lhs_target, rhs_target, binary_op):
    name = "{}_{}_{}".format(lhs_target, binary_op, rhs_target)
    target_dict = {"u": "source node", "e": "edge", "v": "destination node"}
    lhs_str = target_dict[lhs_target]
    rhs_str = target_dict[rhs_target]
    docstring = r"""Generalized SDDMM function.
    It computes edge features by {op} {lhs} features and {rhs} features.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The {lhs} features.
    y : tensor
        The {rhs} features.
    on_format : int
        Which sparse format to select for compute

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    If the feature shape of two input operands do not match, we first broadcasts the features to a unified
    shape (note that the memory usage will not increase accordingly) and then performs the operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.
    """.format(
        op=binary_op, lhs=lhs_str, rhs=rhs_str
    )

    def func(g, x, y, on_format=_COO):
        return gsddmm(
            g,
            binary_op,
            x,
            y,
            lhs_target=lhs_target,
            rhs_target=rhs_target,
            on_format=on_format,
        )

    func.__name__ = name
    func.__doc__ = docstring
    return func


def _register_sddmm_func():
    """Register sddmm functions"""
    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs != rhs:
            for binary_op in ["add", "sub", "mul", "div", "dot"]:
                func = _gen_sddmm_func(lhs, rhs, binary_op)
                setattr(sys.modules[__name__], func.__name__, func)
                __all__.append(func.__name__)


def copy_u(g, x):
    r"""Generalized SDDMM function that copies source node features to edges.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    x : tensor
        The source node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gsddmm(g, "copy_lhs", x, None)


def copy_v(g, x):
    r"""Generalized SDDMM function that copies destination node features to edges.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    x : tensor
        The destination node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gsddmm(g, "copy_rhs", None, x)


_register_sddmm_func()
