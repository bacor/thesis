# Experiment BNG02: Convergence of the Bayesian Naming Game 
# 
# This experiment aims to verify if the Bayesian Naming Game 
# indeed converges.

args0='--runs 5 --T 10000000 --firstrun 6'
args1='--N 20 --K 40 --b 1'
args2='--pi gappy --beta 40'
args3="--zeta 1  --eta 1 --gamma 10000000 --hazard deterministic --chain 0"
args4='--datapoints 500 --datascale log --record 0 --recordutterances 0'
args="$args0 $args1 $args2 $args3 $args4"

# Folders
src="../../src"
out="../../results/BNG02"

echo "Starting experiment"
python $src/BayesianNamingGame.py $args --name BNG02 --out $out
echo "Experiment done"
echo ""
echo "--------------------------------------------"
echo ""