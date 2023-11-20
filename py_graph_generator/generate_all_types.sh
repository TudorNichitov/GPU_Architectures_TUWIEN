conda init bash
conda activate gpu_env

dataDir="../data/all_types/"

python3 main.py -t=erdos -n=10 -p=0.1 --out=$dataDir
python3 main.py -t=erdos -n=100 -p=0.1 --out=$dataDir
python3 main.py -t=erdos -n=1000 -p=0.1 --out=$dataDir
python3 main.py -t=erdos -n=10000 -p=0.001 --out=$dataDir

python3 main.py -t=scale -n=10 --out=$dataDir
python3 main.py -t=scale -n=100 --out=$dataDir
python3 main.py -t=scale -n=1000 --out=$dataDir
python3 main.py -t=scale -n=10000 --out=$dataDir

python3 main.py -t=path -n=10 --out=$dataDir
python3 main.py -t=path -n=100 --out=$dataDir
python3 main.py -t=path -n=1000 --out=$dataDir
python3 main.py -t=path -n=10000 --out=$dataDir

python3 main.py -t=ring -n=10 --out=$dataDir
python3 main.py -t=ring -n=100 --out=$dataDir
python3 main.py -t=ring -n=1000 --out=$dataDir
python3 main.py -t=ring -n=10000 --out=$dataDir

python3 main.py -t=complete -n=10 --out=$dataDir
python3 main.py -t=complete -n=100 --out=$dataDir
python3 main.py -t=complete -n=1000 --out=$dataDir
#python3 main.py -t=complete -n=10000 --out=$dataDir # too dense

python3 main.py -t=tree -n=10 --out=$dataDir
python3 main.py -t=tree -n=100 --out=$dataDir
python3 main.py -t=tree -n=1000 --out=$dataDir
python3 main.py -t=tree -n=10000 --out=$dataDir

python3 main.py -t=empty -n=10 --out=$dataDir
python3 main.py -t=empty -n=100 --out=$dataDir
python3 main.py -t=empty -n=1000 --out=$dataDir
python3 main.py -t=empty -n=10000 --out=$dataDir

python3 main.py -t=custom -n=10 --density=0.2 --out=$dataDir
python3 main.py -t=custom -n=100 --density=0.2 --out=$dataDir
python3 main.py -t=custom -n=1000 --density=0.2 --out=$dataDir
python3 main.py -t=custom -n=10000 --density=0.001 --out=$dataDir
