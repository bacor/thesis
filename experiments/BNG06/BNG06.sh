# Experiment BNG06: Search the entire parameter space
# 
# In this experiment we run explore the entire parameter
# space (gamma, eta, zeta). For all parameter-settings,
# we repeat the game only a few times. 

args0='--runs 10 --T 100000 --firstrun 1'
args1='--N 10 --K 20 --b 1'
args2='--pi gappy --beta 20'
args3="--hazard deterministic --chain 1"
args4='--datapoints 500 --datascale log --record 1 --recordutterances 1'
args="$args0 $args1 $args2 $args3 $args4"

# Parameters
gammas=(1 inf)
etas=(1 inf)
zetas=(1 inf)

for gamma in "${gammas[@]}"
do
	for eta in "${etas[@]}"
	do
		for zeta in "${zetas[@]}"
		do
			echo "Starting experiment $name"
			
			# Setup source, output dir, params, name and go!
			src="../../src"
			out="../../results/BNG06/gamma-$gamma/eta-$eta/"
			params="--gamma $gamma --eta $eta --zeta $zeta"
			name="BNG06-gamma-$gamma-eta-$eta-zeta-$zeta"
			mkdir -p $out
			python $src/BayesianNamingGame.py $args $params --name $name --out $out
			echo "--------------------------------------------"
			echo ""
		done
	done
done