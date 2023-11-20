# Graph Pathfinding: Transitive Closure & Algorithms using CUDA

### Authors: Fenz, Fuchssteiner, Nichitov, Spitzer

##Introduction

This project explores algorithms for determining paths in directed graphs, focusing on the concept of transitive closure - an augmented graph where each vertex is directly connected to every other reachable vertex.

### Graph Representation

Format: Line-based file format for input and output.
Adjacency Matrices: Used internally to represent the graph.

##Approaches

1)Matrix Multiplication
        Multiplies the adjacency matrix up to |V| times.

        Complexity: Time O(|V|^4), Space O(|V|^2).

        Efficiency: Inefficient for larger graphs.

2)Floyd-Warshall Algorithm
        Adapted for reachability, originally for shortest path computation.

        Complexity: Time O(|V|^3), Space O(|V|^2).

3)APSP using SSSP
        Solves All-Pairs Shortest Path by applying Single-Source Shortest Path algorithm.

        Complexity: Time O(|V|^2 log |V| + |V||E|), Space O(|V| + |E|).

        Efficiency: Memory-efficient.

##Implementations

1)CPU: Sequential Floyd-Warshall as a baseline.

2)GPU: Various implementations, including:

       a Matrix Multiplication

       b Naïve Floyd-Warshall (variations with pinned, zero-copy, pitched memory)

       c Tiled Floyd-Warshall with Shared-Memory


## Overview

## Installation
In the following sections we describe how to install & run the application.
Note: The input graph generation is described in **[py_graph_generator/README.md](py_graph_generator/README.md)**

### Prerequisites
1. cuda tool kit (for details go to: https://developer.nvidia.com/cuda-toolkit)
2. cmake (for details go to: https://cmake.org/install/)

### Compile & Run - using shell script
For convenience reasons, we provide a bash script file named *build.sh* to build the application.
Run with:
```bash
  bash build.sh
```
After the execution, a build folder is created in which the executable is stored. 

To test the application with predefined program args run:
```bash
  bash run_single.sh
```

### Compile & Run - manual approach
### Install
Alternatively to the above described shell script compilation, you can build the application manually as follows:
```bash
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=<type> (Release/Debug)
    make
```

### Run
To run the application execute:
```bash
  ./ass2 "kernel" "graphfile" ["outpath"] ["datapath"] [verbosity] [exportGraph]
```

**kernel**: choose from ['cpu', 'kernel1', ..., 'kernel8'] \
**graphfile**: in the datafolder we provide graph files usable as input. \
**outpath**: directory in which the results (out.csv + output_graph file) will be stored. \
**datapath**: directory containing the input graph files \
**verbosity**: 0: no output, 1: info output, 2: debug output
**exportGraph**: 0: skip export to file, 1: export out-graph-file
e.g.:
```bash
  ./ass2 "kernel3" "graph_custom_n_100_density_02_s_42" "../out/" "../data/" 2 1
```

## Results
After executing the program with specific input arguments, the resulting output is stored in the defined 'outpath'.
For small graphs we output additional information to the console. (e.g. adjacency matrix for input and output graph).

out.csv: This file contains the durations of each step a specific kernel needs to compute its result. \
out_graph_*: These files contain the computed transitive closure of the corresponding input graph file.
 
## Reproducibility
### Input Data
The input data must be created with:
```bash
  cd py_graph_generator/
```
```bash
  bash generate_rnd_n_100-1k_samples10.sh
```
```bash
  bash generate_rnd_n_1k-10k_samples5.sh
```

### Run Tests
To reproduce the results shown in the report we provide two bash scripts to run each problem instance 
generated in the previous step on each implementation. 
```bash
  cd ..  # go repo root
```

```bash
  bash run_100_to_1k.sh
```
Large problem instances are only solved by the top three implementations.
```bash
  bash run_1k_to_10k.sh
```

Additionally, we provide the raw test results in [out/1k-10k/out.csv](out/1k-10k/out.csv) 
and [out/100-1k/out.csv](out/100-1k/out.csv). Note: New results will get appended to existing csv files.
- - -
