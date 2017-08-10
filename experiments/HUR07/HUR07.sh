# Experiment HUR06
# 
# This experiment investigates the effect initial conditions for base 7
# 

args='--eta 1 --B 10 --res 200 --timesteps 5001 --runs 300 --agents 200 --simplify 1'

for nmax in {11..110}
do
	echo "Working on $nmax"
	python ../../src/MultiplicativeBaseGame.py $args --nmax $nmax --name HUR07-nmax-$nmax
done

