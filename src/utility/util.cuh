#ifndef ASS2_UTIL_CUH
#define ASS2_UTIL_CUH

#include <sys/time.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <ctime>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CY "\033[33m" // yellow
#define CBY "\033[1m\033[33m" // bold yellow
#define CEND "\033[0m" // reset

namespace util {
    int verbosity = 1;
    /// To store timestamps for predefined events
    struct TimeStamps {
        double start = 0;
        double malloc = 0;
        double memCopy = 0;
        double memSet = 0;
        double kernel = 0;
        double memCopyBack = 0;
        double free = 0;
        double stop = 0;
    };

    bool fileExists(const std::string &csvName) {
        bool exists = false;
        std::ifstream f;
        f.open(csvName);
        if (f) exists = true;
        f.close();
        return exists;
    }

    bool createDir(const std::string &dirName) {
        // Creating a dirName
        if (fileExists(dirName)) return false;
        if (mkdir(dirName.c_str(), 0777) == -1) {
            std::cerr << "Warning: Cannot create dirName: '" << dirName << "' - " << strerror(errno) << std::endl;
            return false;
        }
        return true;
    }

    /// prints matrix to console (only suited for very small graphs)
    void printMatrix(const char *graph, const int &nNodes, const std::string &name) {
        if (verbosity < 2) return;
        std::cout << name << "\n";
        for (int i = 0; i < nNodes; i++) {
            for (int j = 0; j < nNodes; j++) {
                if (graph[i * nNodes + j]) std::cout << CBY << int(graph[i * nNodes + j]) << CEND << "  ";
                else std::cout << int(graph[i * nNodes + j]) << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        std::cout << std::flush;
    }

    /// loads data from graph file and returns adjacency matrix in 1D array
    char *loadGraphData(const std::string &graphFile, int &nNodes, int &nEdges) {
        if (!fileExists(graphFile))
            throw std::invalid_argument("Graph File does not exist!");
        std::ifstream infile(graphFile);
        std::string tag;
        int isUndirected;
        infile >> tag >> nNodes >> nEdges >> isUndirected;

        // init inputGraph as flat array
        char *inputGraph = new char[nNodes * nNodes];
        memset(inputGraph, 0, nNodes * nNodes * sizeof(char));

        // add edges to flat array line by line
        int u, v, weight;
        while (infile >> tag >> u >> v >> weight) inputGraph[v + nNodes * u] = 1;

        infile.close();
        return inputGraph;
    }

    /// stores result adjacency matrix to graph file
    void storeGraphData(const std::string &graphFile, const char *outGraph, const int &nNodes, const int &nEdges) {
        std::ofstream outfile(graphFile);
        std::string tag = "H";
        int isUndirected = 0;
        outfile << tag << " " << nNodes << " " << nEdges << " " << isUndirected << std::endl;

        tag = "E";
        int u, v, weight = 1;
        for (v = 0; v < nNodes; v++) {
            for (u = 0; u < nNodes; u++) {
                if (!outGraph[u + nNodes * v]) continue;
                outfile << tag << " " << u << " " << v << " " << weight << std::endl;
            }
        }
        if (verbosity > 1) std::cout << "Result written to: " << graphFile << std::endl;
        outfile.close();
    }

    /// return Timestamp in ms
    double getTimeMs() {
        struct timeval tp{};
        struct timezone tzp{};
        gettimeofday(&tp, &tzp);
        auto us = ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
        return us * 1000;
    }

    /// computes the elapsed milli seconds using timestamps
    void computeElapsed(const TimeStamps &timestamps, double &eMalloc, double &eMemCopy,
                        double &eMemSet, double &eKernel, double &eMemCopyBack, double &eFree, double &eGPU) {
        eMalloc = std::max(timestamps.malloc - timestamps.start, 0.0);
        eMemCopy = std::max(timestamps.memCopy - timestamps.malloc, 0.0);
        eMemSet = std::max(timestamps.memSet - timestamps.memCopy, 0.0);
        eKernel = std::max(timestamps.kernel - timestamps.memSet, 0.0);
        eMemCopyBack = std::max(timestamps.memCopyBack - timestamps.kernel, 0.0);
        eFree = std::max(timestamps.free - timestamps.memCopyBack, 0.0);
        eGPU = std::max(timestamps.stop - timestamps.start, 0.0);
    }

    /// prints elapsed times
    void printTimings(const std::string &solverName, TimeStamps timestamps) {
        // Compute Timings
        double eMalloc, eMemCopy, eMemSet, eKernel, eMemCopyBack, eFree, eTotal;
        computeElapsed(timestamps, eMalloc, eMemCopy, eMemSet, eKernel, eMemCopyBack, eFree, eTotal);

        if (verbosity > 1) printf("Elapsed time %s%s%s:\n", CY, solverName.c_str(), CEND);
        if (verbosity > 1 && eMalloc > 0) printf("\tMalloc %.2f ms\n", eMalloc);
        if (verbosity > 1 && eMemCopy > 0) printf("\tMemCopy %.2f ms\n", eMemCopy);
        if (verbosity > 1 && eMemSet > 0) printf("\tMemSet %.2f ms\n", eMemSet);
        if (verbosity > 1 && eKernel > 0) printf("\tKernel: %s%.2f ms%s\n", CY, eKernel, CEND);
        if (verbosity > 1 && eMemCopyBack > 0) printf("\tMemCopyBack %.2f ms\n", eMemCopyBack);
        if (verbosity > 1 && eFree > 0) printf("\tFree %.2f ms\n", eFree);
        if (verbosity > 0 && eTotal > 0) printf("\tTotal: %s%.2f ms%s \n", CBY, eTotal, CEND);
        if (verbosity > 1) printf("\n");
    }

    /// Creates csv file and writes header if file not exist
    void createCSV(const std::string &csvFile) {
        bool exists = fileExists(csvFile);
        std::fstream outfile;
        outfile.open(csvFile, std::fstream::out | std::fstream::app);
        if (!exists) {
            // if file not exists create file and write header
            outfile << "Created,"
                    << "solverName,"
                    << "graphFileName,"
                    << "eMalloc,"
                    << "eMemCopy,"
                    << "eMemSet,"
                    << "eKernel,"
                    << "eMemCopyBack,"
                    << "eFree,"
                    << "eTotal"
                    << std::endl;
        }
        outfile.close();
    }

    void outputCurrentTime(const tm *now, std::fstream &outfile) {
        outfile << now->tm_year + 1900 << "-" << now->tm_mon + 1 << "-" << now->tm_mday
                << " " << now->tm_hour << ":" << now->tm_min << ",";
    }

    /// Writes time measurements to csv file
    void writeTsLineToCsv(const std::string &csvFile, const std::string &kernel,
                          const std::string &graphFile, TimeStamps ts) {
        double eMalloc, eMemCopy, eMemSet, eKernel, eMemCopyBack, eFree, eGPU;
        computeElapsed(ts, eMalloc, eMemCopy, eMemSet, eKernel, eMemCopyBack, eFree, eGPU);
        createCSV(csvFile);

        std::time_t t = std::time(nullptr); // get time now.
        std::tm *now = std::localtime(&t);

        std::fstream outfile;
        outfile.open(csvFile, std::ios_base::app);
        outputCurrentTime(now, outfile);
        outfile << kernel << ","
                << graphFile << ","
                << eMalloc << ","
                << eMemCopy << ","
                << eMemSet << ","
                << eKernel << ","
                << eMemCopyBack << ","
                << eFree << ","
                << eGPU << std::endl;
        outfile.close();

        if (verbosity > 1) std::cout << "Added entry in: " << csvFile << std::endl;
    }

    void setVerbosity(int verbosityLevel) {
        verbosity = verbosityLevel;
    }
}

#endif  //ASS2_UTIL_CUH