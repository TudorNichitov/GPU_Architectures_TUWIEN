#ifndef ASS2_KERNEL8_THRUST_CUH
#define ASS2_KERNEL8_THRUST_CUH

#include <cctype>
#include <cuda_runtime.h>
#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>


namespace gpu8 {
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

        if (x < size && !adjM[x])
            adjM[x] = (char) (adjM[u + nNodes * v] && adjM[w + nNodes * u]);
    }


    util::TimeStamps solve(const int &nNodes, const char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;
        const int matrixSize = nNodes * nNodes;

        // Initialize the grid and block dimensions here
        int blockDimX = 32 * 8;
        dim3 dimGrid((matrixSize - 1) / blockDimX + 1, 1, 1);
        dim3 dimBlock(blockDimX, 1, 1);

        // Initialize the inputGraph as thrust vector
        // Does not count towards total time because in real-life applications
        // there would be no intermediate format to begin with
        thrust::host_vector<char> h_inputGraph(matrixSize);
        thrust::copy(inputGraph, inputGraph + matrixSize, h_inputGraph.begin());


        ts.start = util::getTimeMs();

        // Allocate Mem and copy to Device
        thrust::device_vector<int> d_nEdges(1);
        thrust::device_vector<char> d_inputGraph(h_inputGraph.begin(), h_inputGraph.end());

        ts.malloc = util::getTimeMs();

        ts.memCopy = util::getTimeMs();

        // Set nEdges to 0 on device
        d_nEdges[0] = 0;
        ts.memSet = util::getTimeMs();

        // Compute on device
        for (int u = 0; u < nNodes; ++u) {
            naiveFwKernel<<<dimGrid, dimBlock>>>(u, nNodes, thrust::raw_pointer_cast(d_inputGraph.data()));
        }
        countEdges<<<dimGrid, dimBlock>>>(nNodes, thrust::raw_pointer_cast(d_inputGraph.data()),
                                          thrust::raw_pointer_cast(d_nEdges.data()));
        CHECK(cudaDeviceSynchronize())
        ts.kernel = util::getTimeMs();

        // Copy back
        nEdges = d_nEdges[0];
        thrust::copy(d_inputGraph.begin(), d_inputGraph.end(), h_inputGraph.begin());

        ts.memCopyBack = util::getTimeMs();

        // Free mem on device
        ts.free = util::getTimeMs();
        ts.stop = util::getTimeMs();

        // Conversion back from thrust format to original
        thrust::copy(h_inputGraph.begin(), h_inputGraph.end(), outputGraph);
        return ts;
    }
}

#endif //ASS2_KERNEL8_THRUST_CUH