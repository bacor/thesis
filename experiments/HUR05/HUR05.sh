# Experiment HUR04
# 
# This experiment investigates the effect initial conditions for base 7
# 

args='--eta 1 --B 10 --res 10 --timesteps 25000 --runs 300 --agents 200'

for freq in 1 1.5 2 3
do
	python ../../src/AdditiveBaseGame.py $args --initbase 8 --initfreq $freq --name HUR05-base-8-freq-$freq
done
