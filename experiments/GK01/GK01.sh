#PBS -S /bin/bash
#PBS -lnodes=1:cores16
#PBS -lwalltime=12:00:00

# Load module and move to right directory
module load node
cd ~/webppl-il

# Requires
REQS="--require ~/webppl-il --require webppl-fs --require webppl-timeit"
#export REQS

# Run in parallel
parallel --gnu  ~/node_modules/webppl/webppl GK01.wppl "n=10000 b={1} alpha={2} eps={3} $REQS" ::: {1..10} ::: 0.01 0.5 ::: 0.001 0.05
