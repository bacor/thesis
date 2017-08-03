# Experiment HUR02
# 
# In this experiment we run Hurfords first basic simulation 
# (with additive agents) for various values of B
# 

args='--eta 1 --res 10 --timesteps 40 --runs 300 --agents 200 '

for B in 1 5 10 15 20 25 30
do
	python ../../src/AdditiveBaseGame.py $args --B $B --name HUR01-B-$B --out results
done
