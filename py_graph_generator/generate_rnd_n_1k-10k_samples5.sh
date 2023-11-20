conda init bash
conda activate gpu_env

seed=2349184
nSamples=5
dataDir="../data/1k-10k/"

nNodesValues=()
nNodesValues+=(1000)
nNodesValues+=(2000)
nNodesValues+=(3000)
nNodesValues+=(4000)
nNodesValues+=(5000)
nNodesValues+=(6000)
nNodesValues+=(7000)
nNodesValues+=(8000)
nNodesValues+=(9000)
nNodesValues+=(10000)

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
