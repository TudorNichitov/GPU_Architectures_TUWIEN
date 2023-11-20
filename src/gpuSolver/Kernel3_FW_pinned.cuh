#ifndef ASS2_KERNEL3_FW_PINNED_CUH
#define ASS2_KERNEL3_FW_PINNED_CUH

#include <cctype>
#include <cuda_runtime.h>
#include <cstdint>


namespace gpu3 {
    __global__ void countEdges(const int nNodes, char *outputGraph, int *nEdges) {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= nNodes * nNodes)
            return;

        if (x / nNodes == x % nNodes) // add reflective edges
            outputGraph[x] = 1;

        if (x < nNodes * nNodes)
            atomicAdd(nEdges, outputGraph[x]);

    }

    __global__ void naiveFwKernel(const int u, const int nNodes, char *adjM) {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        const auto size = nNodes * nNodes;
        const auto v = x / nNodes;
        const auto w = x % nNodes;

        if (x < size && !adjM[x]) {
            adjM[x] = (char) (adjM[u + nNodes * v] && adjM[w + nNodes * u]);
        }
    }

    util::TimeStamps solve(const int &nNodes, const char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;

        const auto matrixSize = nNodes * nNodes * sizeof(char);
        char *d_outMatrix;
        char *h_outMatrix;  // used as pinned memory
        int *d_NEdges;

        // Initialize the grid and block dimensions here
        int blockDimX = 32 * 8;
        dim3 dimGrid((nNodes * nNodes - 1) / blockDimX + 1, 1, 1);
        dim3 dimBlock(blockDimX, 1, 1);

        ts.start = util::getTimeMs();

        // Allocate Mem on Device
        CHECK(cudaMallocHost(&h_outMatrix, matrixSize))
        CHECK(cudaMalloc(&d_outMatrix, matrixSize))
        CHECK(cudaMalloc(&d_NEdges, sizeof(int)))
        ts.malloc = util::getTimeMs();

        // Copy to Device
//        memcpy(h_outMatrix, inputGraph, matrixSize);
        CHECK(cudaMemcpy(h_outMatrix, inputGraph, matrixSize, cudaMemcpyHostToHost))
        CHECK(cudaMemcpy(d_outMatrix, h_outMatrix, matrixSize, cudaMemcpyHostToDevice))

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
        CHECK(cudaMemcpy(h_outMatrix, d_outMatrix, matrixSize, cudaMemcpyDeviceToHost))

//        memcpy(outputGraph, h_outMatrix, matrixSize);
        ts.memCopyBack = util::getTimeMs();
        CHECK(cudaMemcpy(outputGraph, h_outMatrix, matrixSize, cudaMemcpyHostToHost))

        // Free mem on device
        CHECK(cudaFree(d_outMatrix))
        CHECK(cudaFree(d_NEdges))
        CHECK(cudaFreeHost(h_outMatrix))
        ts.free = util::getTimeMs();

        ts.stop = util::getTimeMs();
        return ts;
    }
}

#endif //ASS2_KERNEL3_FW_PINNED_CUH