#include <stdexcept>
#include "utility/util.cuh"
#include "cpuSolver/cpu1_FW.cuh"
#include "gpuSolver/Kernel1_MatrixMulti.cuh"
#include "gpuSolver/Kernel2_FW_naive.cuh"
#include "gpuSolver/Kernel3_FW_pinned.cuh"
#include "gpuSolver/Kernel4_FW_zeroCopy.cuh"
#include "gpuSolver/Kernel5_FW_pitched.cuh"
#include "gpuSolver/Kernel6_FW_pitched2Dblock.cuh"
#include "gpuSolver/Kernel7_tiled_FW_sharedMem.cuh"
#include "gpuSolver/Kernel8_thrust.cuh"
#include "gpuSolver/Kernel9_thrust_SSSP.cuh"

using namespace std;

const char *DEFAULT_KERNEL = "kernel5";
const char *DEFAULT_GRAPH_FILE = "graph_path_n_10";
const char *DEFAULT_DATA_PATH = "../data/";
const char *DEFAULT_OUT_PATH = "../out/";
const int DEFAULT_VERBOSITY = 2; // 0: no output, 1: info, 2: debug
const bool DEFAULT_GRAPH_EXPORT = false; // true: transitive closure is exported to file, false: no export

/// run with
/// ass2 "kernel" "graphfile" "outpath" "datapath" "verbosity" "skipGraphExport"
int main(int argc, char **argv) {
    // parse args
    std::string solverName(DEFAULT_KERNEL);
    std::string graphFile(DEFAULT_GRAPH_FILE);
    std::string outdir(DEFAULT_OUT_PATH);
    std::string datadir(DEFAULT_DATA_PATH);
    int verbosity(DEFAULT_VERBOSITY);
    bool exportGraph = DEFAULT_GRAPH_EXPORT;

    if (argc >= 2) solverName = argv[1];
    if (argc >= 3) graphFile = argv[2];
    if (argc >= 4) outdir = argv[3];
    if (argc >= 5) datadir = argv[4];
    if (argc >= 6) verbosity = stoi(argv[5]);
    if (argc >= 7) exportGraph = stoi(argv[6]);

    std::string inGraphFile = datadir + graphFile;
    util::createDir(outdir);
    std::string csvFile = outdir + "out.csv";
    std::string outGraphFile = outdir + "out_" + graphFile + "_" + solverName;
    util::setVerbosity(verbosity);

    // load input
    int nNodes;
    int nEdges;
    char *inputGraph = util::loadGraphData(inGraphFile, nNodes, nEdges);

    // init output variables
    int outNEdges = nEdges;
    auto outputGraph = new char[nNodes * nNodes];
    memset(outputGraph, 0, nNodes * nNodes * sizeof(char));

    // start computation
    util::TimeStamps ts;
    if (solverName == "cpu1") { // naive Floyd Warshall (FW)
        ts = cpu1::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel1") { // matrix multiplication
        ts = gpu1::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel2") { // naive FW
        ts = gpu2::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel3") { // naive FW pinned memory
        ts = gpu3::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel4") { // naive FW zero-copy memory
        ts = gpu4::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel5") { // naive FW pitched
        ts = gpu5::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel6") { // naive FW pitched 2D block/grid
        ts = gpu6::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel7") { // tiled FW shared memory
        ts = gpu7::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel8") { // thrust library
        ts = gpu8::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else if (solverName == "kernel9") { // thrust library with SSSP
        ts = gpu9::solve(nNodes, inputGraph, outputGraph, outNEdges);
    } else throw std::invalid_argument("Solver not found! Choose from ['cpu1', ..., cpuN, 'kernel1', ..., kernelM]");

    // result handling
    util::printTimings(solverName, ts);
    util::writeTsLineToCsv(csvFile, solverName, graphFile, ts);

    if (verbosity > 1) {
        printf("-- Edges in output graph: %d\n", outNEdges);

        if (nNodes <= 20) { // output adjacency matrices for small graphs only
            util::printMatrix(inputGraph, nNodes, "Input");
            util::printMatrix(outputGraph, nNodes, "Output");
        }
    }

    // export transitive closure graph to file
    if (exportGraph) {
        util::storeGraphData(outGraphFile, outputGraph, nNodes, outNEdges);
    }
    delete[] outputGraph;
    delete[] inputGraph;
    if (verbosity > 0) std::cout << "---" << std::endl << std::endl;
    return 0;
}