import argparse
import os

import dgl
import numpy as np
import pandas
import torch

from dgl.data.utils import generate_mask_tensor


def get_rand_type():
    val = np.random.uniform()
    if val < 0.1:
        return 0
    elif val < 0.4:
        return 1
    return 2


def gen_graph_from_text(txt_path: str, to_bidirected: bool = False):
    df = pandas.read_csv(
        txt_path, sep="\t", skiprows=4, header=None, names=["src", "dst"]
    )
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    g = dgl.graph((src, dst))
    if to_bidirected:
        g = dgl.to_bidirected(g)
    g = dgl.to_simple(g)
    g = dgl.compact_graphs(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    print("generating mask begin")
    num_nodes = g.num_nodes()
    node_types = np.array([get_rand_type() for i in range(num_nodes)])
    g.ndata["node_type"] = torch.tensor(node_types, dtype=torch.int32)
    train_mask = node_types == 0
    val_mask = node_types == 1
    test_mask = node_types == 2
    print("generating mask done")
    print("generating mask train mask tensor")
    g.ndata["train_mask"] = generate_mask_tensor(train_mask)
    print("generating mask train val tensor")
    g.ndata["val_mask"] = generate_mask_tensor(val_mask)
    print("generating mask train test tensor")
    g.ndata["test_mask"] = generate_mask_tensor(test_mask)
    g.ndata.pop("node_type")
    g.ndata.pop("_ID")

    print("saving graph...")
    parent_folder = os.path.dirname(txt_path)
    graph_name = os.path.basename(txt_path).split(".")[0]
    save_path = os.path.join(parent_folder, f"{graph_name}.bin")
    dgl.save_graphs(save_path, [g])
    print("saving graph done")
    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which dataset are you going to use?")
    parser.add_argument(
        "--txt-path", type=str, help="The path of the txt file", required=True
    )
    parser.add_argument("--to-bidirected", action="store_true")
    args = parser.parse_args()
    to_bidirected = args.to_bidirected is True

    graph = gen_graph_from_text(args.txt_path, to_bidirected=to_bidirected)
    print(graph.nodes())
    print(graph)
    print("transfer txt graph successful")
