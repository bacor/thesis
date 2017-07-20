# Experiment BNG03: Find out the result of parameters on convergence
# 
# In this experiment we analyze the effect of different parameters
# on convergence time

args0='--runs 20 --T 1000000 --firstrun 1'
#args1='--N 5 --K 20 --b 10'
args2='--beta 100'
args3="--zeta 1 --eta 1 --gamma inf --hazard deterministic --chain 0"
args4='--datapoints 500 --datascale log --record 0 --recordutterances 0'
args="$args0 $args2 $args3 $args4"

# Folders
src="../../src"
out="../../results/BNG03"

for N in 5 50 100 150 200
do
	echo "Starting experiment with N=$N"
	python $src/BayesianNamingGame.py $args --N $N --K 20 --b 10 --name BNG03-N-$N --out $out
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done


for K in 5 50 100 150 200
do
	echo "Starting experiment with K=$K"
	python $src/BayesianNamingGame.py $args --N 5 --K $K --b 10 --name BNG03-K-$K --out $out
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done


for b in 1 2 5 10 20 50 100
do
	echo "Starting experiment with b=$b"
	python $src/BayesianNamingGame.py $args --N 5 --K 20 --b $b --name BNG03-b-$b --out $out
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done