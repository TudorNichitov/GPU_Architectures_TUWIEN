#ifndef ASS2_KERNEL2_FW_NAIVE_CUH
#define ASS2_KERNEL2_FW_NAIVE_CUH

#include <cctype>
#include <cuda_runtime.h>
#include <cstdint>


namespace gpu2 {
    __global__ void countEdges(const int nNodes, char *adjM, int *nEdges) {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= nNodes * nNodes)
            return;

        if (x / nNodes == x % nNodes) // add reflective edges
            adjM[x] = 1;

        if (x < nNodes * nNodes)
            atomicAdd(nEdges, adjM[x]);
    }

    __global__ void naiveFwKernel(const int u, const int nNodes, char *adjM) {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        const auto size = nNodes * nNodes;
        const auto v = x / nNodes;
        const auto w = x % nNodes;

        if (x < size && !adjM[x]) {
            adjM[x] = (char)(adjM[u + nNodes * v] && adjM[w + nNodes * u]);
        }
    }

    util::TimeStamps solve(const int &nNodes, const char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;

        const auto matrixSize = nNodes * nNodes * sizeof(char);
        char *d_outMatrix;
        int *d_NEdges;

        // Initialize the grid and block dimensions here
        int blockDimX = 32 * 8;
        dim3 dimGrid((nNodes * nNodes - 1) / blockDimX + 1, 1, 1);
        dim3 dimBlock(blockDimX, 1, 1);

        ts.start = util::getTimeMs();

        // Allocate Mem on Device
        CHECK(cudaMalloc(&d_outMatrix, matrixSize))
        CHECK(cudaMalloc(&d_NEdges, sizeof(int)))
        ts.malloc = util::getTimeMs();

        // Copy to Device
        CHECK(cudaMemcpy(d_outMatrix, inputGraph, matrixSize, cudaMemcpyHostToDevice))
        ts.memCopy = util::getTimeMs();

        // Set nEdges to 0 on device
        CHECK(cudaMemset(d_NEdges, 0, sizeof(int)))
        ts.memSet = util::getTimeMs();

        // Compute on device
        for (int u = 0; u < nNodes; ++u) {
            naiveFwKernel<<<dimGrid, dimBlock>>>(u, nNodes, d_outMatrix);
        }
        countEdges<<<dimGrid, dimBlock>>>(nNodes, d_outMatrix, d_NEdges);
        CHECK(cudaDeviceSynchronize())
        ts.kernel = util::getTimeMs();

        // Copy back
        CHECK(cudaMemcpy(&nEdges, d_NEdges, sizeof(int), cudaMemcpyDeviceToHost))
        CHECK(cudaMemcpy(outputGraph, d_outMatrix, matrixSize, cudaMemcpyDeviceToHost))
        ts.memCopyBack = util::getTimeMs();

        // Free mem on device
        CHECK(cudaFree(d_outMatrix))
        CHECK(cudaFree(d_NEdges))
        ts.free = util::getTimeMs();

        ts.stop = util::getTimeMs();
        return ts;
    }
}

#endif //ASS2_KERNEL2_FW_NAIVE_CUH