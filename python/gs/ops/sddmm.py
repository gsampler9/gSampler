from itertools import product
import sys

from .sparse import gsddmm as gsddmm_internal
from ..format import _COO, _CSC, _CSR

__all__ = ["gsddmm"]


def reshape_lhs_rhs(lhs_data, rhs_data):
    r"""Expand dims so that there will be no broadcasting issues with different
    number of dimensions. For example, given two shapes (N, 3, 1), (E, 5, 3, 4)
    that are valid broadcastable shapes, change them to (N, 1, 3, 1) and
    (E, 5, 3, 4)

    Parameters
    ----------
    lhs_data : tensor or None
        The left operand, could be None if it's not required by op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by op.
    """
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
    r"""Generalized Sampled-Dense-Dense Matrix Multiplication interface.
    It computes edge features by :attr:`op` lhs features and rhs features.

    .. math::

        x_{e} = \phi(x_{lhs}, x_{rhs}), \forall (u,e,v)\in \mathcal{G}

    where :math:`x_{e}` is the returned feature on edges and :math:`x_u`,
    :math:`x_v` refers to :attr:`u`, :attr:`v` respectively. :math:`\phi`
    is the binary operator :attr:`op`, and :math:`\mathcal{G}` is the graph
    we apply gsddmm on: :attr:`g`. :math:`lhs` and :math:`rhs` are one of
    :math:`u,v,e`'s.

    Parameters
    ----------
    g : gs.Matrix
        The input graph.
    op : str
        Binary operator, could be ``add``, ``sub``, ``mul``, ``div``, ``dot``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by op.
    lhs_target: str
        Choice of ``u``(source), ``e``(edge) or ``v``(destination) for left operand.
    rhs_target: str
        Choice of ``u``(source), ``e``(edge) or ``v``(destination) for right operand.

    Returns
    -------
    tensor
        The result tensor.
    """
    if op not in ["copy_lhs", "copy_rhs"]:
        lhs_data, rhs_data = reshape_lhs_rhs(lhs_data, rhs_data)
    return gsddmm_internal(
        g._graph, op, lhs_data, rhs_data, lhs_target, rhs_target, on_format
    )


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
