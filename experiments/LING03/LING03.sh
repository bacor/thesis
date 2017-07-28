# Experiment LING03
# 
# Lateral inhibition name for various values of d_inc

args='--agents 200 --timesteps 30000 --limit 1 --res 100 --runs 300'


src='../../src'
out='results/'
for dinc in 0 1 10 50 100 1000
do
	echo "Starting experiment with d-inc=$dinc"
	params="--dinit 1 --ddec 0 --dinh 1 --dinc $dinc"
	python $src/LateralInhibitionNamingGame.py $params $args --name LING03-dinc-$dinc --out $out
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done
