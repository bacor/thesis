# Experiment BNG07: Demonstrate the effect of beta
# 
# In this experiment we extend experint 04 by explicitly illustrating
# the effect of beta for a single prior (peak)

args0='--runs 15 --T 1000000 --firstrun 1'
args1='--N 10 --K 20 --b 10'
args2='--pi peak'
args3="--zeta 1 --eta 1 --gamma 10000000 --hazard deterministic --chain 0"
args4='--datapoints 500 --datascale log --record 0 --recordutterances 0'
args="$args0 $args1 $args2 $args3 $args4"

# Folders
src="../../src"
out="../../results/BNG07"


for beta in 10 20 500 1000
do
	echo "Starting experiment with beta=$beta"
	python $src/BayesianNamingGame.py $args --beta $beta --name BNG07-beta-$beta --out $out
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done