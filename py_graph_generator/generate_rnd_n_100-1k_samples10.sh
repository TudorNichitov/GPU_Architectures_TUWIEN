conda init bash
conda activate gpu_env

seed=94256
nSamples=10
dataDir="../data/100-1k"

nNodesValues=()
nNodesValues+=(100)
nNodesValues+=(200)
nNodesValues+=(300)
nNodesValues+=(400)
nNodesValues+=(500)
nNodesValues+=(600)
nNodesValues+=(700)
nNodesValues+=(800)
nNodesValues+=(900)
nNodesValues+=(1000)

densityValues=()
densityValues+=(.001)
densityValues+=(.005)
densityValues+=(.01)
densityValues+=(.05)
densityValues+=(.1)

pValues=()
pValues+=(.001)
pValues+=(.005)
pValues+=(.01)
pValues+=(.05)
pValues+=(.1)

for nNodes in ${nNodesValues[*]}; do
  # custom graph (assignment requirement) multiple densities
  for density in ${densityValues[*]}; do
    python3 main.py -t=custom -n=$nNodes --density=$density -x=$nSamples --seed=$seed --out=$dataDir
  done

  # erdös renyi graph multiple edge probabilities
  for p in ${pValues[*]}; do
    python3 main.py -t=erdos -n=$nNodes -p=$p -x=$nSamples --seed=$seed --out=$dataDir
  done

  # scale free graph
  python3 main.py -t=scale -n=$nNodes -x=$nSamples --seed=$seed --out=$dataDir

  # random tree graph
  python3 main.py -t=tree -n=$nNodes -x=$nSamples --seed=$seed --out=$dataDir
done
