# GPU Parallelization of 2-Body Distance Histograms (CUDA)

## Overview

<img width="368" height="200" alt="image" align="right" src="https://github.com/user-attachments/assets/a59d1853-3e02-44a3-a0c0-d5d546707689" />

This project implements a CUDA kernel for computing pairwise (2-body) distance histograms over a set of 3D points. The implementation focuses on applying memory-aware parallelization strategies and load-balancing techniques to efficiently compute all point-to-point distances while avoiding double counting.

The work is inspired by established GPU algorithms for 2-body statistics and explores how different parallelization strategies interact with shared memory usage, atomic operations, and control divergence.

<br clear="right">

## Implementation Summary

### Tiling
Point data is divided into tiles assigned to CUDA blocks. Each block:
- Loads its local point data into registers
- Loads point data from a second block into shared memory
- Performs inter-block and intra-block distance computations

This approach reduces global memory traffic and increases data reuse during distance calculations.

### Output Privatization
To reduce contention from atomic updates to a global histogram:
- Each block maintains a private histogram in shared memory
- After all comparisons are complete, private histograms are merged into the global output

An advanced variant using multiple private histograms per block was also explored but was not retained due to increased shared memory overhead without consistent performance benefit.

### Load Balancing
Two forms of load balancing are applied to reduce control divergence and uneven workload distribution:

- **Intrablock balancing**  
  Threads within a block are assigned a balanced number of point comparisons instead of following a triangular comparison pattern.

- **Interblock balancing**  
  Blocks are assigned a balanced subset of block-to-block comparisons rather than decreasing workloads for higher-index blocks.

## Reference
This implementation draws inspiration from the following work:

[1] N. Pitaksirianan, Z. N. Lewis, and Y.-C. Tu, “Algorithms and framework for computing 2-body statistics on GPUs,” *Distributed and Parallel Databases*, vol. 37, no. 4, pp. 587–622, Aug. 2018. https://doi.org/10.1007/s10619-018-7238-0
