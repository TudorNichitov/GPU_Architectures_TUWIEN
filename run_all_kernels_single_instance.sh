#!/bin/bash

bash build.sh
cd build || exit

datapath="../data/"
outpath="../out/small/"
graphfile="graph_path_n_10"

verbosity=2
exportGraph=1

kernels=()
kernels+=("cpu1")
kernels+=("kernel1")
kernels+=("kernel2")
kernels+=("kernel3")
kernels+=("kernel4")
kernels+=("kernel5")
kernels+=("kernel6")
kernels+=("kernel7")
kernels+=("kernel8")
kernels+=("kernel9")



for kernel in ${kernels[*]}; do
  echo "./ass2 $kernel $graphfile $outpath $datapath $verbosity $exportGraph"
  ./ass2 "$kernel" "$graphfile" "$outpath" "$datapath" "$verbosity" "$exportGraph"
done
