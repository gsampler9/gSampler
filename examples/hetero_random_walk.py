from typing import List
import gs
import torch


def HeteroRandomWalk(HA: gs.HeteroMatrix, seeds: torch.Tensor, metapath: List):
    ret = [seeds, ]
    for etype in metapath:
        A = HA.get_homo_matrix(etype)
        subA = A.fused_columnwise_slicing_sampling(seeds, 1, True)
        seeds = subA.row_indices(False)
        ret.append(seeds)
    return torch.stack(ret)


def HeteroRandomWalkFused(HA: gs.HeteroMatrix, seeds: torch.Tensor, metapath: List):
    return HA.metapath_random_walk(seeds, metapath)
