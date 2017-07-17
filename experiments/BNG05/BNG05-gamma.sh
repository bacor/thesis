# BNG05-gamma
#
# The purpose of experiment BNG05 is to explore the parameter space
# of the Bayesian Naming Game. In this subexperiment we vary zeta, 
# while keeping all other variables constant.
zeta=1.5

args0='--runs 30'
args1='--N 20 --K 40 --b 1 --T 100000'
args2='--pi gappy --beta 1'
args3="--zeta $zeta --hazard deterministic --chain 1"
args4='--datapoints 500 --datascale log'
args="$args0 $args1 $args2 $args3 $args4"

for gamma in 1 5 10 20 50 100 1000 100000000
do
	echo "Starting experiment with gamma = $gamma"
	name="	BNG05-zeta-$zeta"
	python ../BayesianNamingGame.py $args --gamma $gamma --name $name --out results/gamma-$gamma
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done