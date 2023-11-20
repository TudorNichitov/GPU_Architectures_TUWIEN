#!/bin/bash

bash build.sh
cd build || exit

kernel="kernel2"            # choose from ['cpu1', 'kernel1', ..., 'kernel9']
datapath="../data/"         # directory containing graph files (Note! trailing '/' is mandatory)
graphfile="graph_path_n_10" # graph file name
outpath="../out/single/"    # out directory for out-graph-file & out.csv (will be created)
verbosity=2                 # 0: no output, 1: info output, 2: debug output
exportGraph=1               # 0: skip export to file, 1: export out-graph-file

echo "./ass2 $kernel $graphfile $outpath $datapath $verbosity $exportGraph"
./ass2 "$kernel" "$graphfile" "$outpath" "$datapath" "$verbosity" "$exportGraph"
