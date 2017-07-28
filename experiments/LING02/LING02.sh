# Experiment LING02
# 
# Lateral inhibition name for various values of d_inh

args='--agents 200 --timesteps 20000 --limit 1 --res 100 --runs 300'


src='../../src'
out='results/'

# values: [0] + [4**i for i in range(-4,3)]
for dinh in 0 0.00390625 0.015625 0.0625 0.25 1 4 16
do
	echo "Starting experiment with d-inh=$dinh"
	params="--dinit 1 --ddec 0 --dinc 1 --dinh $dinh"
	python $src/LateralInhibitionNamingGame.py $params $args --name LING02-dinh-$dinh --out $out
	echo "Experiment done"
	echo ""
	echo "--------------------------------------------"
	echo ""
done
