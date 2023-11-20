#ifndef ASS2_KERNEL1_MATRIXMULTI_CUH
#define ASS2_KERNEL1_MATRIXMULTI_CUH
// Naive Matrix multiplication approach
// Given the adjacency matrix A, calculate A + A^2 + A^3 + ... + A^n -> All non-zero elements are part of the reachability matrix

#include <cctype>
#include <cuda_runtime.h>
#include <cstdint>

namespace gpu1 {
    __global__ void matrixMultiplicationKernel(const char *lastStep, const char *adjacencyMatrix, char *nextStep,
                                               char *reachability, const int nodes, int *nEdges, char *hasChanged) {
        const auto row = blockIdx.y * blockDim.y + threadIdx.y;
        const auto col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row == col && row < nodes) {
            reachability[row * nodes + col] = 1;
            atomicAdd(nEdges, 1);
            return;
        }

        char tmpSum = 0;
        if (row < nodes && col < nodes) {
            // Calculate one element of A^n
            for (int i = 0; i < nodes; i++) {
                tmpSum = (char) (tmpSum || adjacencyMatrix[row * nodes + i] && lastStep[i * nodes + col]);
            }
            if (tmpSum != lastStep[row * nodes + col]) {
                hasChanged[0] = 1;
            }
            nextStep[row * nodes + col] = tmpSum;

            // Add this element of A^n to reachability matrix. Use logical or because we're only interested if we reach or not
            if (tmpSum) reachability[row * nodes + col] = 1;

            // Count the number of edges in the resulting graph
            if (reachability[row * nodes + col]) atomicAdd(nEdges, 1);
        }
    }

    // Given a reachability matrix after n steps, calculate A^(n+1) and add it to the reachability matrix
    void performStep(char *lastStep, char *adjacencyMatrix, char *nextStep, char *reachability,
                     const int nodes, int *nEdges, char *hasChanged) {
        CHECK(cudaMemcpy(lastStep, nextStep, nodes * nodes * sizeof(char), cudaMemcpyDeviceToDevice))
        dim3 block(nodes, nodes);
        dim3 grid(1, 1);
        if (nodes * nodes > 512) { // Can't have more than 512 threads in a block
            block.x = 32;
            block.y = 32;
            grid.x = ceil(double(nodes) / double(block.x));
            grid.y = ceil(double(nodes) / double(block.y));
        }

        CHECK(cudaMemset(hasChanged, 0, sizeof(char)))

        matrixMultiplicationKernel<<<grid, block>>>(
                adjacencyMatrix, lastStep, nextStep, reachability, nodes, nEdges, hasChanged);
        CHECK(cudaDeviceSynchronize())
    }

    util::TimeStamps solve(const int &nNodes, const char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;
        const auto matrixSize = nNodes * nNodes * sizeof(char);

        char *devAdjMatrix;      // Original adjacency matrix on device
        char *devCurStepMatrix;  // A^n ... where n is the current step
        char *devNextStep;       // A^(n+1) .. where n is the current step
        char *devReachMatrix;    // A + A^2 + .. + A^n - Reachability matrix for current step
        int *devNEdges;
        char *devHasChanged;
        char stepChange;

        // Allocate & Copy Mem to Device
        ts.start = util::getTimeMs();

        // Allocate memory on GPU
        CHECK(cudaMalloc((void **) &devAdjMatrix, matrixSize))
        CHECK(cudaMalloc((void **) &devCurStepMatrix, matrixSize))
        CHECK(cudaMalloc((void **) &devNextStep, matrixSize))
        CHECK(cudaMalloc((void **) &devReachMatrix, matrixSize))
        CHECK(cudaMalloc((void **) &devNEdges, sizeof(int)))
        CHECK(cudaMalloc((void **) &devHasChanged, sizeof(char)))
        ts.malloc = util::getTimeMs();

        CHECK(cudaMemcpy(devAdjMatrix, inputGraph, matrixSize, cudaMemcpyHostToDevice))
        CHECK(cudaMemcpy(devNextStep, devAdjMatrix, matrixSize, cudaMemcpyDeviceToDevice))
        CHECK(cudaMemcpy(devReachMatrix, devNextStep, matrixSize, cudaMemcpyDeviceToDevice))
        ts.memCopy = util::getTimeMs();

        // No need for any memsets
        ts.memSet = util::getTimeMs();

        for (int i = 1; i < nNodes; i++) {
            CHECK(cudaMemset(devNEdges, 0, sizeof(int)))
            performStep(devCurStepMatrix, devAdjMatrix, devNextStep, devReachMatrix, nNodes, devNEdges, devHasChanged);
            CHECK(cudaMemcpy(&stepChange, devHasChanged, sizeof(char), cudaMemcpyDeviceToHost))

            if (!stepChange) break;
        }
        ts.kernel = util::getTimeMs();

        CHECK(cudaMemcpy(&nEdges, devNEdges, sizeof(int), cudaMemcpyDeviceToHost))
        CHECK(cudaMemcpy(outputGraph, devReachMatrix, matrixSize, cudaMemcpyDeviceToHost))
        ts.memCopyBack = util::getTimeMs();

        CHECK(cudaFree(devAdjMatrix))
        CHECK(cudaFree(devReachMatrix))
        CHECK(cudaFree(devCurStepMatrix))
        CHECK(cudaFree(devNextStep))
        CHECK(cudaFree(devNEdges))
        ts.free = util::getTimeMs();
        ts.stop = util::getTimeMs();

        return ts;
    }
}

#endif //ASS2_KERNEL1_MATRIXMULTI_CUH