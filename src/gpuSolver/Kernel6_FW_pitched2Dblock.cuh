#ifndef ASS2_KERNEL6_FW_PITCHED2DBLOCK_CUH
#define ASS2_KERNEL6_FW_PITCHED2DBLOCK_CUH

#include <cctype>
#include <cuda_runtime.h>
#include <cstdint>


namespace gpu6 {
    __global__ void countEdges(const int nNodes, const size_t nMemoryRow, char *adjM, int *nEdges) {
        const auto  v = blockDim.x * blockIdx.x + threadIdx.x;
        const auto  w = blockDim.y * blockIdx.y + threadIdx.y;
        const auto  wv = w * nMemoryRow + v;

        if (v < nNodes && w < nNodes) {
            if (v == w) adjM[wv] = 1; // add reflective edges
            atomicAdd(nEdges, adjM[wv]);
        }
    }

    __global__ void pitchedFwKernel(const int u, const size_t nMemoryRow, const int nNodes, char *outputGraph) {
        const auto  v = blockDim.x * blockIdx.x + threadIdx.x;
        const auto  w = blockDim.y * blockIdx.y + threadIdx.y;

        const auto  wv = w * nMemoryRow + v;
        const auto  uv = u * nMemoryRow + v;
        const auto  wu = w * nMemoryRow + u;

        if (v < nNodes && w < nNodes && !outputGraph[wv])
            outputGraph[wv] = (char)(outputGraph[wu] && outputGraph[uv]);
    }

    util::TimeStamps solve(const int &nNodes, const char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;

        char *d_outMatrix;
        int *d_NEdges;

        // Initialize the grid and block dimensions here
        int blockDimX = 16;
        dim3 dimGrid((nNodes - 1) / blockDimX + 1, (nNodes - 1) / blockDimX + 1, 1);
        dim3 dimBlock(blockDimX, blockDimX, 1);
        ts.start = util::getTimeMs();

        // Allocate Mem on Device
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
        auto nMemoryRow = pitch / sizeof(char); // sizeof(char) redundant -> to emphasize: pitch is in bytes
        for (int u = 0; u < nNodes; ++u) {
            pitchedFwKernel<<<dimGrid, dimBlock>>>(u, nMemoryRow, nNodes, d_outMatrix);
        }
        countEdges<<<dimGrid, dimBlock>>>(nNodes, nMemoryRow, d_outMatrix, d_NEdges);
        CHECK(cudaDeviceSynchronize())
        ts.kernel = util::getTimeMs();

        // Copy back
        CHECK(cudaMemcpy2D(outputGraph, nNodes, d_outMatrix, pitch, nNodes * sizeof(char), nNodes,
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

#endif //ASS2_KERNEL6_FW_PITCHED2DBLOCK_CUH