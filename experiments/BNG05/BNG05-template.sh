# BNG05
#
# The purpose of experiment BNG05 is to explore the parameter space
# of the Bayesian Naming Game. The default parameters are:
# 
# - runs: 30
#
# - N = 20
# - K = 40
# - b = 1
# - T = 100 000
# 
# - pi: 'gappy'
# - beta: 1
# 
# - zeta: 1
# - gamma: 1e8
# - hazard: deterministic
# - chain: true
# 
# - datapoints: 500
# - datascale: log
# 
# 
# In every simulation we vary only a single parameter

args0='--runs 30 --out results'
args1='--N 20 --K 40 --b 1 --T 100000'
args2='--pi gappy --beta 1'
args3='--zeta 1 --gamma 100000000 --hazard deterministic --chain 1'
args4='--datapoints 500 --datascale log'
args="$args0 $args1 $args2 $args3 $args4"

for param in value value
do
	echo "Starting experiment with zeta = $zeta"
	name="BNG05-"
	python ../BayesianNamingGame.py --param $param $args --name $name
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done