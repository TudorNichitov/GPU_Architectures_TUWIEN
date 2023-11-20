#ifndef ASS2_KERNEL9_THRUST_CUH
#define ASS2_KERNEL9_THRUST_CUH

#include <cctype>
#include <cuda_runtime.h>
#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>


#define thrustPtr(a) thrust::raw_pointer_cast(a.data())

// Implements the algorithm presented in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.4206&rep=rep1&type=pdf
// which performs a SSSP (Single Source Shortest Path) algorithm on each node sequentially.
// Performs worse on smaller graphs but much better on bigger and sparser graphs because it requires less memory
namespace gpu9 {

    // Initializes the M, C and U vectors to 0 except for the currently processed vertex which is set to 1
    __global__ void ssspInit(const int nNodes, const int S, char *M, char *C) {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= nNodes) return;

        C[x] = (char) (x == S);
        M[x] = (char) (x == S);
    }

    // If the given vertex is set in M, checks for all neighbors whether they have already been marked as reachable.
    // If previously unreachable, they are marked reachable in U and the Mupdated flag is set
    __global__ void ssspKernel1(const int nNodes, const int *Edst, const int *Edge_idx, char *M, char *C, char &Mupdated) {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= nNodes) return;

        if (M[x]) {
            M[x] = 0;

            int lastIdx = Edge_idx[x + 1];

            for (int i = Edge_idx[x]; i < lastIdx; i++) {
                const int dst = Edst[i];
                if (!C[dst]) {
                    Mupdated = 1;
                    C[dst] = 1;
                    M[dst] = 1;
                }
            }
        }
    }

    util::TimeStamps solve(const int &nNodes, const char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;

        // Initialize the grid and block dimensions here
        int blockDimX = 32 * 8;
        dim3 dimGrid((nNodes - 1) / blockDimX + 1, 1, 1);
        dim3 dimBlock(blockDimX, 1, 1);

        // Transform adjacency matrix into alternative representation (edge vector)
        // Store the destination vector of each edge
        thrust::host_vector<int> h_Edst;

        // For each vector, store the index of the first outgoing edge, last entry is set to the total number of edges to have boundary for last vector
        thrust::host_vector<int> h_Edge_idx(nNodes + 1);

        nEdges = 0;
        for (int x = 0; x < nNodes; x++) {
            h_Edge_idx[x] = nEdges;
            for (int y = 0; y < nNodes; y++) {
                if (inputGraph[x * nNodes + y]) {
                    nEdges++;
                    h_Edst.push_back(y);
                }
            }
        }
        h_Edge_idx[nNodes] = nEdges;

        ts.start = util::getTimeMs();

        // Allocate Mem and copy to Device
        thrust::device_vector<int> d_Edst(h_Edst.begin(), h_Edst.end());

        // For each node, store the index at which its edges start, last idx is equal to the the number of edges
        // to have an end index for the last node
        thrust::device_vector<int> d_Edge_idx(h_Edge_idx.begin(), h_Edge_idx.end());

        thrust::copy(h_Edst.begin(), h_Edst.end(), d_Edst.begin());
        thrust::copy(h_Edge_idx.begin(), h_Edge_idx.end(), d_Edge_idx.begin());

        // C = Cost vector -> 0 = Unreachable, 1 = Reachable
        // U = Updated Cost vector for intermediate results
        // M = Mask to indicate when the costs for a vertice have to be recalculated
        thrust::device_vector<char> d_C(nNodes);
        thrust::device_vector<char> d_U(nNodes);
        thrust::device_vector<char> d_M(nNodes);

        // Flag to indicate when no update was done in last iteration
        thrust::device_vector<char> d_Mupdated(1);

        ts.malloc = util::getTimeMs();

        ts.memCopy = util::getTimeMs();

        ts.memSet = util::getTimeMs();

        // Compute on device

        nEdges = 0;
        for (int u = 0; u < nNodes; ++u) {
            ssspInit<<<dimGrid, dimBlock>>>(nNodes, u, thrustPtr(d_M), thrustPtr(d_C));
            CHECK(cudaDeviceSynchronize());

            do {
                d_Mupdated[0] = 0;
                ssspKernel1<<<dimGrid, dimBlock>>>(nNodes, thrustPtr(d_Edst), thrustPtr(d_Edge_idx), thrustPtr(d_M),
                                                   thrustPtr(d_C), *thrustPtr(d_Mupdated));
                CHECK(cudaDeviceSynchronize())
            } while (d_Mupdated[0]);

            thrust::copy(d_C.begin(), d_C.end(), &(outputGraph[u * nNodes]));
            nEdges += thrust::count(d_C.begin(), d_C.end(), 1);
        }

        ts.kernel = util::getTimeMs();

        // Copy back
        ts.memCopyBack = util::getTimeMs();

        // Free mem on device
        ts.free = util::getTimeMs();
        ts.stop = util::getTimeMs();

        // Conversion back from thrust format to original
        return ts;
    }
}

#endif //ASS2_KERNEL9_THRUST_CUH