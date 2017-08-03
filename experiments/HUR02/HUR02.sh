# Experiment HUR02
# 
# In this experiment we check the effect of eta, fixing B=10
# and keeping all other parameters identical to HUR01, except
# for the timesteps
# 
args='--B 10 --res 10 --timesteps 15000 --runs 300 --agents 200'

for eta in 1 2 3
do
	python ../../src/AdditiveBaseGame.py $args --eta $eta --name HUR02-eta-$eta
done
