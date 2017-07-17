# BNG05-zeta
#
# The purpose of experiment BNG05 is to explore the parameter space
# of the Bayesian Naming Game. In this subexperiment we vary zeta, 
# while keeping all other variables constant.

args0='--runs 30 --out results/gamma-100000000'
args1='--N 20 --K 40 --b 1 --T 100000'
args2='--pi gappy --beta 1'
args3='--gamma 100000000 --hazard deterministic --chain 1'
args4='--datapoints 500 --datascale log'
args="$args0 $args1 $args2 $args3 $args4"

for zeta in 10 #1 1.5 2 2.5 3 4 5 10 20 50 100
do
	echo "Starting experiment with zeta = $zeta"
	name="	BNG05-zeta-$zeta"
	python ../BayesianNamingGame.py $args --zeta $zeta --name $name
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done