#!/bin/bash

bash build.sh
cd build || exit

datapath10k="../data/1k-10k/"
datapath5k="../data/1k-5k/"
outpath="../out/1k-10k/"
verbosity=0
exportGraph=0

kernels=()
kernels+=("kernel7")
kernels+=("kernel8")
kernels+=("kernel9")

# fast CUDA kernels are evaluated on 1k to 10k
for fullgraphfile in "${datapath10k}"graph*; do
  graphfile=$(basename "$fullgraphfile")
  for kernel in ${kernels[*]}; do
    echo "./ass2 $kernel $graphfile $outpath $datapath10k $verbosity $exportGraph"
    ./ass2 "$kernel" "$graphfile" "$outpath" "$datapath10k" "$verbosity" "$exportGraph"
  done
done

# cpu implementation is only evaluated till to 5k
for fullgraphfile in "${datapath5k}"graph*; do
  graphfile=$(basename "$fullgraphfile")
  echo "./ass2 cpu1 $graphfile $outpath $datapath5k $verbosity $exportGraph"
  ./ass2 "$kernel" "$graphfile" "$outpath" "$datapath5k" "$verbosity" "$exportGraph"
done