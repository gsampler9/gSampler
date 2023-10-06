# gSampler: Efficient GPU-Based Graph Sampling for Graph Learning

This repository contains the source code for the SOSP23 paper titled ["gSampler: General and Efficient GPU-based Graph Sampling for Graph Learning"](https://dl.acm.org/doi/10.1145/3600006.3613168).

`gSampler` is a high-performance GPU-based graph sampling technology specifically designed for graph learning. It utilizes the "Extract-Compute-Select-Finalize" (ECSF) model for single-layer graph sampling and provides matrix-centric APIs that are user-friendly and intuitive.

To optimize the application of various sampling algorithms on different datasets, `gSampler` incorporates data flow graphs and introduces several optimizations, including:

* `Computation optimizations`: Eliminating redundant data movements and operators by optimizing the data flow graph.
* `Data layout optimizations`: Determining the most efficient data storage format for executing the program based on the sampling algorithm and dataset.
* `Super-batching`: Batching the sampling tasks to improve GPU utilization without compromising training accuracy.

Compared to existing solutions, `gSampler` offers unprecedented programming support and performance improvements for graph sampling, particularly for training graph neural networks (GNNs).

# Repository Directory Structure

The repository is structured as follows:

```
├── CMakeLists.txt
├── examples/            # Usage examples for gSampler
├── LICENSE
├── python/
│   ├── gs/
│   │   ├── format.py
│   │   ├── jit/
│   │   │   ├── module.py   # Main compilation module (gs.jit.compile)
│   │   │   ├── optimize.py # Implementation of optimization passes (e.g., CSE, kernel fusion)
│   │   │   └── trace.py    # Modified torch.fx for graph sampling
│   │   ├── matrix_api/
│   │   │   ├── batch_matrix.py # BatchMatrix API with super-batching
│   │   │   └── matrix.py       # Matrix API
│   │   ├── ops/           # Python wrapper for C++ API
│   │   └── utils/
│   └── setup.py
├── readme.md
├── scripts/
├── src/                   # Implementation of low-level C++ API for matrices
├── tests/
└── third_party/
```

# Installation

## Software Version
* g++/gcc >= v9.4.0
* CMake >= v3.16.3
* CUDA >= v11.6
* PyTorch >= v1.12.1
* DGL >= v0.9.1

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

4. Testing the Package Path
   ```shell
   python3
   import gs
   print(gs.package_path)
   # Expecting output in this section
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


For more details, refer to the `example` folder. You can test each demo by running `python *.py` directly. Additionally, the code for reproducing the evaluations mentioned in the paper is available in this [repository](https://github.com/gpzlx1/gsampler-artifact-evaluation).

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
# At this step, you need to compile the `graphsage_sampler` function using the given arguments. 
# Note that it will convert List, Integer, and Float into constants during compilation.
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


# Datasets

This repository currently supports two types of native datasets: `Reddit` and all `OGB Node Property Prediction Datasets`. You can access them using the functions `gs.utils.load_reddit` and `gs.utils.load_ogb` for downloading and preprocessing.

To work with other datasets, follow these steps:

1. Prepare the graph in CSC format.
2. Load the dataset using the `m.load_graph` API.

```python
# Prepare the graph in CSC format
csc_indptr, csc_indices = load_graph(...)

# Load the graph into GPU memory
m = gs.Matrix()
m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

# For large-scale graphs with Unified Virtual Addressing (UVA)
m.load_graph("CSC", [csc_indptr.pin_memory(), csc_indices.pin_memory()])

# Utilize super-batching by converting Matrix to BatchMatrix
bm = gs.BatchMatrix()
bm.load_from_matrix(m)
```



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

# Citing gSampler
If you find this repo helpful, please cite the paper.
```tex
@inproceedings{10.1145/3600006.3613168,
author = {Gong, Ping and Liu, Renjie and Mao, Zunyao and Cai, Zhenkun and Yan, Xiao and Li, Cheng and Wang, Minjie and Li, Zhuozhao},
title = {GSampler: General and Efficient GPU-Based Graph Sampling for Graph Learning},
year = {2023},
isbn = {9798400702297},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3600006.3613168},
doi = {10.1145/3600006.3613168},
abstract = {Graph sampling prepares training samples for graph learning and can dominate the training time. Due to the increasing algorithm diversity and complexity, existing sampling frameworks are insufficient in the generality of expression and the efficiency of execution. To close this gap, we conduct a comprehensive study on 15 popular graph sampling algorithms to motivate the design of gSampler, a general and efficient GPU-based graph sampling framework. gSampler models graph sampling using a general 4-step Extract-Compute-Select-Finalize (ECSF) programming model, proposes a set of matrix-centric APIs that allow to easily express complex graph sampling algorithms, and incorporates a data-flow intermediate representation (IR) that translates high-level API codes for efficient GPU execution. We demonstrate that implementing graph sampling algorithms with gSampler is easy and intuitive. We also conduct extensive experiments with 7 algorithms, 4 graph datasets, and 2 hardware configurations. The results show that gSampler introduces sampling speedups of 1.14--32.7\texttimes{} and an average speedup of 6.54\texttimes{}, compared to state-of-the-art GPU-based graph sampling systems such as DGL, which translates into an overall time reduction of over 40\% for graph learning. gSampler is open-source at https://tinyurl.com/29twthd4.},
booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles},
pages = {562–578},
numpages = {17},
keywords = {graph neural network, graphics processing unit, graph learning, graph sampling},
location = {Koblenz, Germany},
series = {SOSP '23}
}
```
