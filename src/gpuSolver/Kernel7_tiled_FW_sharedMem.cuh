/// Implementation of the approach proposed in Katz, Gary J., and Joseph T. Kider.
/// "All-pairs shortest-paths for large graphs on the GPU." (2008): 47.

#ifndef ASS2_KERNEL7_TILED_FW_SHAREDMEM_CUH
#define ASS2_KERNEL7_TILED_FW_SHAREDMEM_CUH

#include <cctype>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE_GPU7 16
namespace gpu7 {
    /// naive approach to sum up edges from adjacency matrix
    __global__ void countEdges(const int nNodes, const size_t nMemoryRow, char *adjM, int *nEdges) {
        const auto v = blockDim.x * blockIdx.x + threadIdx.x;
        const auto w = blockDim.y * blockIdx.y + threadIdx.y;
        const auto wv = w * nMemoryRow + v;

        if (v < nNodes && w < nNodes) {
            if (v == w) adjM[wv] = 1; // add reflective edges
            atomicAdd(nEdges, adjM[wv]);
        }
    }

    /// pattern approach to sum up edges from adjacency matrix using shared memory
    __global__ void countEdgesPattern(const int nNodes, const size_t nMemoryRow, char *adjM, int *nEdges) {
        const auto v = BLOCK_SIZE_GPU7 * blockIdx.y + threadIdx.y;
        const auto w = BLOCK_SIZE_GPU7 * blockIdx.x + threadIdx.x;
        const auto sharedMemSize = BLOCK_SIZE_GPU7 * BLOCK_SIZE_GPU7;

        __shared__ int sharedAdjM[sharedMemSize];
        const auto cellId = v * nMemoryRow + w;
        const auto sharedId = threadIdx.y * BLOCK_SIZE_GPU7 + threadIdx.x;

        // add reflective edges
        if (v == w && v < nNodes) adjM[cellId] = 1;

        // copy matrix to shared
        sharedAdjM[sharedId] = (int) ((v < nNodes && w < nNodes) && adjM[cellId]);

        // sum shared mem with pattern
#pragma unroll
        for (int stride = sharedMemSize / 2; stride; stride >>= 1) {
            if (sharedId < stride) {
                __syncthreads();
                sharedAdjM[sharedId] += sharedAdjM[sharedId + stride];
            } else return;
        }

        // add partial result to global
        if (sharedId == 0)
            atomicAdd(nEdges, sharedAdjM[0]);
    }

    /// Updates Primary Block only (i.e. solve primary block as sub problem)
    __global__ void phase1(const int primBlockId, const size_t nMemoryRow, const int nNodes, char *adjM) {
        const auto x = threadIdx.x;
        const auto y = threadIdx.y;

        const auto v = BLOCK_SIZE_GPU7 * primBlockId + y;
        const auto w = BLOCK_SIZE_GPU7 * primBlockId + x;

        // init shared mem
        __shared__ bool sharedAdjM[BLOCK_SIZE_GPU7][BLOCK_SIZE_GPU7];
        const auto cellId = v * nMemoryRow + w;
        sharedAdjM[y][x] = (v < nNodes && w < nNodes) && (bool) adjM[cellId];

        // Synchronize to ensure that sharedAdjM is initialized completely => done within for loop
        __syncthreads();

        bool newEdge;
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE_GPU7; ++u) {
            newEdge = sharedAdjM[y][u] && sharedAdjM[u][x];
            __syncthreads();  // sync block before each iteration
            sharedAdjM[y][x] = sharedAdjM[y][x] || newEdge;
        }

        // copy back to global
        if (v < nNodes && w < nNodes) {
            adjM[cellId] = (char) sharedAdjM[y][x];
        }
    }

    /// Updates blocks orthogonal to primary block dependent on primary block (precomputed in phase 1)
    /// blockIdx.y in {0,1}. 0: blocks in primary row, 1: blocks in primary col
    __global__ void phase2(const int primBlockId, const size_t nMemoryRow, const int nNodes, char *adjM) {
        if (blockIdx.x == primBlockId) return;
        // STEP 1: load primary block into shared
        const auto x = threadIdx.x;
        const auto y = threadIdx.y;

        auto v = BLOCK_SIZE_GPU7 * primBlockId + y;
        auto w = BLOCK_SIZE_GPU7 * primBlockId + x;

        __shared__ bool sharedAdjMPrim[BLOCK_SIZE_GPU7][BLOCK_SIZE_GPU7];
        const auto primGlobalId = v * nMemoryRow + w;
        sharedAdjMPrim[y][x] = v < nNodes && w < nNodes && (bool) adjM[primGlobalId];

        // STEP 2: load orthogonal block into shared
        //     - Update either v or w index to access orthogonal blocks only (global Mem index is adjusted accordingly)
        __shared__ bool sharedAdjMSecondary[BLOCK_SIZE_GPU7][BLOCK_SIZE_GPU7];
        if (blockIdx.y == 0) w = BLOCK_SIZE_GPU7 * blockIdx.x + x;
        else v = BLOCK_SIZE_GPU7 * blockIdx.x + y;

        //     - load orthogonal block into shared
        const auto secGlobId = v * nMemoryRow + w;
        sharedAdjMSecondary[y][x] = (v < nNodes && w < nNodes) && (bool) adjM[secGlobId];

        // STEP 3: update orthogonal block dependent on primary block
        bool hasEdge;
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE_GPU7; ++u) {
            __syncthreads();
            hasEdge = blockIdx.y == 0 ?
                      sharedAdjMPrim[y][u] && sharedAdjMSecondary[u][x] :
                      sharedAdjMSecondary[y][u] && sharedAdjMPrim[u][x];
            sharedAdjMSecondary[y][x] = sharedAdjMSecondary[y][x] || hasEdge;
        }

        // write data to global mem
        if (v < nNodes && w < nNodes) {
            adjM[secGlobId] = (char) sharedAdjMSecondary[y][x];
        }
    }

    /// Update ternary blocks
    __global__ void phase3(const int primBlockId, const size_t nMemoryRow, const int nNodes, char *adjM) {
        if (blockIdx.x == primBlockId || blockIdx.y == primBlockId) return;

        const auto x = threadIdx.x;
        const auto y = threadIdx.y;

        const auto v = BLOCK_SIZE_GPU7 * blockIdx.y + y;
        const auto w = BLOCK_SIZE_GPU7 * blockIdx.x + x;

        __shared__ bool sharedAdjRow[BLOCK_SIZE_GPU7][BLOCK_SIZE_GPU7];
        __shared__ bool sharedAdjCol[BLOCK_SIZE_GPU7][BLOCK_SIZE_GPU7];

        const auto wCol = BLOCK_SIZE_GPU7 * primBlockId + x;
        const auto vRow = BLOCK_SIZE_GPU7 * primBlockId + y;

        // Load data for block
        sharedAdjRow[y][x] = (vRow < nNodes && w < nNodes) && (bool) adjM[vRow * nMemoryRow + w];
        sharedAdjCol[y][x] = (v < nNodes && wCol < nNodes) && (bool) adjM[v * nMemoryRow + wCol];

        if (!(v < nNodes && w < nNodes)) return;

        // Compute data for block
        const auto cellId = v * nMemoryRow + w;
        bool hasEdge = adjM[cellId];

#pragma unroll
        for (int u = 0; u < BLOCK_SIZE_GPU7; ++u) {
            __syncthreads();
            hasEdge = hasEdge || (sharedAdjCol[y][u] && sharedAdjRow[u][x]);
        }
        adjM[cellId] = (char) hasEdge;
    }

    util::TimeStamps solve(const int &nNodes, const char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;

        char *d_outMatrix;
        int *d_NEdges;

        // Initialize grid and block dimensions
        int nBlocks = (nNodes - 1) / BLOCK_SIZE_GPU7 + 1;
        dim3 gridDimP1(1, 1, 1); // grid-size = 1 since only the primary block gets computed
        dim3 gridDimP2(nBlocks, 2, 1); // '2' since we compute each row/column aligned with primary block
        dim3 gridDimP3(nBlocks, nBlocks, 1); // all blocks are computed except primary row/column are idle
        dim3 dimBlock(BLOCK_SIZE_GPU7, BLOCK_SIZE_GPU7, 1);

        ts.start = util::getTimeMs();

        // Allocate Mem on Device current and predecessor AdjMatrices
        size_t pitch;
        CHECK(cudaMallocPitch(&d_outMatrix, &pitch, nNodes * sizeof(char), nNodes))
        CHECK(cudaMalloc(&d_NEdges, sizeof(int)))
        ts.malloc = util::getTimeMs();

        // Copy to Device
        CHECK(cudaMemcpy2D(d_outMatrix, pitch, inputGraph, nNodes, nNodes * sizeof(char), nNodes,
                           cudaMemcpyHostToDevice))
        ts.memCopy = util::getTimeMs();

        // Set nEdges to 0 on device
        CHECK(cudaMemset(d_NEdges, 0, sizeof(int)))
        ts.memSet = util::getTimeMs();

        // Compute on device
        auto nMemoryRow = pitch / sizeof(char);
        for (int primaryBlockId = 0; primaryBlockId < nBlocks; ++primaryBlockId) {
            // phase1: update primary block
            phase1<<<gridDimP1, dimBlock>>>(primaryBlockId, nMemoryRow, nNodes, d_outMatrix);
            // phase2: update secondary block (row/column aligned blocks with primary)
            phase2<<<gridDimP2, dimBlock>>>(primaryBlockId, nMemoryRow, nNodes, d_outMatrix);
            // phase3: update ternary blocks (rest)
            phase3<<<gridDimP3, dimBlock>>>(primaryBlockId, nMemoryRow, nNodes, d_outMatrix);
        }

        countEdgesPattern<<<gridDimP3, dimBlock>>>(nNodes, nMemoryRow, d_outMatrix, d_NEdges);
        CHECK(cudaDeviceSynchronize())
        ts.kernel = util::getTimeMs();

        // Copy back
        CHECK(cudaMemcpy2D(outputGraph, nNodes, d_outMatrix, nMemoryRow, nNodes * sizeof(char), nNodes,
                           cudaMemcpyDeviceToHost))
        CHECK(cudaMemcpy(&nEdges, d_NEdges, sizeof(int), cudaMemcpyDeviceToHost))
        ts.memCopyBack = util::getTimeMs();

        // Free mem on device
        CHECK(cudaFree(d_outMatrix))
        CHECK(cudaFree(d_NEdges))
        ts.free = util::getTimeMs();
        ts.stop = util::getTimeMs();

        return ts;
    }
}

#endif //ASS2_KERNEL7_TILED_FW_SHAREDMEM_CUH