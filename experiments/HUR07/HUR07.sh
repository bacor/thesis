# Experiment HUR06
# 
# This experiment investigates the effect initial conditions for base 7
# 

args='--eta 1 --B 10 --res 200 --timesteps 5001 --runs 150 --agents 200 --simplify 1'

for nmax in {27..34} # {20..100} #36 46 59 74 91 110 52 66 82 100
do
	echo "Working on $nmax"
	python ../../src/MultiplicativeBaseGame.py $args --nmax $nmax --name HUR07-nmax-$nmax
done

