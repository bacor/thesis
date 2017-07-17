# Experiment BNG04: Shape of limiting distribution
# 
# In this experiment we run the game with various hyperparameters alpha
# to see how alpha is reflected in the shape of the limiting distribution

args0='--runs 15 --T 1000000 --firstrun 1'
args1='--N 10 --K 20 --b 10'
args2='--beta 100'
args3="--zeta 1 --eta 1 --gamma 10000000 --hazard deterministic --chain 0"
args4='--datapoints 500 --datascale log --record 0 --recordutterances 0'
args="$args0 $args1 $args2 $args3 $args4"

# Folders
src="../../src"
out="../../results/BNG04"


for pi in flat stair_up stair_down peak lower_half upper_half gappy
do
	echo "Starting experiment with pi=$pi"
	python $src/BayesianNamingGame.py $args --pi $pi --name BNG04-pi-$pi --out $out
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done