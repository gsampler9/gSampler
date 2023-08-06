# gSampler: Efficient GPU-Based Graph Sampling for Graph Learning

`gSampler` is a high-performance GPU-based graph sampling technology specifically designed for graph learning. It utilizes the "Extract-Compute-Select-Finalize" (ECSF) model for single-layer graph sampling and provides matrix-centric APIs that are user-friendly and intuitive.

To optimize the application of various sampling algorithms on different datasets, `gSampler` incorporates data flow graphs and introduces several optimizations, including:

* `Computation optimizations`: Eliminating redundant data movements and operators by optimizing the data flow graph.
* `Data layout optimizations`: Determining the most efficient data storage format for executing the program based on the sampling algorithm and dataset.
* `Super-batching`: Batching the sampling tasks to improve GPU utilization without compromising training accuracy.

Compared to existing solutions, `gSampler` offers unprecedented programming support and performance improvements for graph sampling, particularly for training graph neural networks (GNNs).

# Installation

## Software Version
* Ubuntu 20.04
* g++/gcc v9.4.0
* CMake v3.16.3
* CUDA v11.6
* PyTorch v1.12.1
* DGL v0.9.1

## Install gSampler

To install `gSampler`, use `pip` to manage your Python environment.

1. Install PyTorch and DGL.
2. Download gSampler using the following command:
   ```shell
   git clone --recursive https://github.com/gsampler9/gSampler.git
   ```
3. Build and Install:
   ```shell
   cd gsampler && mkdir build
   cd build && cmake ..
   # compile
   make -j
   # install
   cd ../python && python3 setup.py install
   ```

# Supported algorithms
`gSampler` supports various graph sampling algorithms, including:

* FastGCN
* LADIES
* AS-GCN
* GraphSAGE
* KGCN
* PASS
* ShadowGNN
* GraphSAINT
* RandomWalk
* Node2Vec

Please check the `example` folder for more details. Additionally, the code for reproducing the evaluations mentioned in the paper is available in this [repository](https://github.com/gpzlx1/gsampler-artifact-evaluation).

# Usage

The gSampler is designed specifically for graph sampling. To achieve end-to-end training, it must be integrated with the DGL/PyG Graph Neural Network (GNN) Framework. Here's a minimal demo for the combination:

```python
import gs
import dgl
import torch

# Step 1: prepare graph dataset in csc format
data = dgl.data.RedditDataset(self_loop=True)
g = data[0]
csc_indptr, csc_indices, _ = g.adj_sparse("csc")


# Step 2: Create Matrix
m = gs.Matrix()
## for graph in GPU memory
m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

'''
For large scale graph via Unified Virtual Addressing (UVA)
m.load_graph("CSC", [csc_indptr.pin_memory(), csc_indices.pin_memory()])

To utilize super-batching, please convert Matrix to BatchMatrix
bm = gs.matrix_api.BatchMatrix()
bm.load_from_matrix(m)
'''


# Step 3: Write sampler function with Matrix/BatchMatirx
def graphsage_sampler(A: gs.Matrix, seeds: torch.Tensor,
                      fanouts: List):
    input_node = seeds
    ret = []
    for K in fanouts:
        subA = A[:, seeds]
        sampleA = subA.individual_sampling(K, None, False)
        seeds = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())
    output_node = seeds
    return input_node, output_node, ret


# Step 4: Create Model, Optimizer, dataloader
model = ...
opt = ...
train_seedloader = gs.SeedGenerator(train_nid, batch_size=batch_size)


# Step 5: Compile sampler function with provided arguments
# At this step, you need to compile the `graphsage_sampler` function using the given arguments. Ensure to use constants for Matrix, List, Integer, and Float, but not for tensors.
compile_func = gs.jit.compile(func=graphsage_sampler, args=(m, train_seedloader.data[:batch_size], fanouts))


# Step 6: Execute the training loop. Now, you can proceed with the training loop:
for epoch in range(args.num_epoch):
    model.train()
    for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
        input_node, output_node, blocks = compile_func(m, seeds, fanouts)

        # ... Your training code ...

        loss = model(blocks, ...)

        # ... More training code ...

        opt.zero_grad()
        loss.backward()
        opt.step()
```

For detailed E2E training, please refer `examples/graphsage/graphsage_e2e.py` and `examples/ladies/ladies_e2e.py`


# License

```
Copyright 2023 Ping Gong, Renjie Liu, Zunyao Mao, Zhenkun Cai

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```