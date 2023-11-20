#!/bin/bash

bash build.sh
cd build || exit

datapath="../data/100-1k/"
outpath="../out/100-1k/"
verbosity=0
exportGraph=0

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

for fullgraphfile in "${datapath}"graph*; do
  graphfile=$(basename "$fullgraphfile")
  for kernel in ${kernels[*]}; do
    echo "./ass2 $kernel $graphfile $outpath $datapath $verbosity $exportGraph"
    ./ass2 "$kernel" "$graphfile" "$outpath" "$datapath" "$verbosity" "$exportGraph"
  done
done
