# gSampler
`gSampler` is a general and efficient GPU-based graph sampling technology specifically designed for graph learning. Using the "Extract-Compute-Select-Finalize" (ECSF) model for single-layer graph sampling, gSampler provides matrix-centric APIs that are user-friendly and intuitive.

To effectively accelerate the application of various sampling algorithms on different datasets, `gSampler` introduces data flow graphs to implement various optimizations, including `computation optimizations` (pre-processing, operator fusion, and common passes), `data layout optimizations` (automated format selection and graph compaction), and `super-batching`.
* `Computation optimizations`: eliminate redundant data movements and operators by optimizing on the data flow graph.
* `Data layout optimizations`: determine the most efficient data storage format to execute the program based on the sampling algorithm and dataset.
* `Super-batching`: batch the sampling tasks to improve GPU utilization without affecting training accuracy.

Compared to existing works, `gSampler` provides unprecedented programming support and performance improvement for graph sampling, especially for training graph neural networks (GNNs).

# Install
## Software Version
* Ubuntu 20.04
* g++/gcc v9.4.0
* CMake v3.16.3
* CUDA v11.6
* PyTorch v1.12.1
* DGL v0.9.1

## Install gSampler
We use `pip` to manage our python environment.

1. Install PyTorch and DGL
2. Download gSampler systems
    ```shell
    git clone --recursive https://github.com/gsampler9/gSampler.git
    ```
3. Build and Install
    ```shell
    cd gsampler && mkdir build
    cd build && cmake ..
    # compile
    make -j12
    # install
    cd ../python && python3 setup.py install
    ```

# Supported algorithms
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

See `example` folder for details. More algorithms are being expanded.