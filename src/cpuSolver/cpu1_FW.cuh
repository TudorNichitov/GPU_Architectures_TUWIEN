#ifndef ASS2_CPU1_FW_CUH
#define ASS2_CPU1_FW_CUH

#include <iostream>
#include <numeric>

/// naive Floyd Warshall
namespace cpu1 {
    void floydWarshall(char *outputGraph, int nNodes) {
        // add reflective edges
        for (int u = 0; u < nNodes; u++) {
            outputGraph[u + nNodes * u] = 1;
        }
        // perform floyd warshall
        for (int u = 0; u < nNodes; u++) {
            for (int v = 0; v < nNodes; v++) {
                for (int w = 0; w < nNodes; w++) {
                    if (outputGraph[w + nNodes * v]) continue;
                    outputGraph[w + nNodes * v] = (char) (outputGraph[u + nNodes * v] && outputGraph[w + nNodes * u]);
                }
            }
        }
    }

    void computeTransitiveClosure(const char *inputGraph, const int &nNodes, char *outputGraph, int &nEdges) {
        memcpy(outputGraph, inputGraph, nNodes * nNodes * sizeof(char));
        floydWarshall(outputGraph, nNodes);
        nEdges = std::accumulate(outputGraph, outputGraph + nNodes * nNodes, 0);
    }

    util::TimeStamps solve(const int &nNodes, char *inputGraph, char *outputGraph, int &nEdges) {
        util::TimeStamps ts;
        ts.start = util::getTimeMs();
        computeTransitiveClosure(inputGraph, nNodes, outputGraph, nEdges);
        ts.stop = util::getTimeMs();
        return ts;
    }
}
#endif //ASS2_CPU1_FW_CUH