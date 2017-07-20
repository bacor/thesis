# Experiment BNG06: Search the entire parameter space
# 
# In this experiment we run explore the entire parameter
# space (gamma, eta, zeta). For all parameter-settings,
# we repeat the game only a few times. 

args0='--runs 10 --T 100000 --firstrun 1'
args1='--N 10 --K 20 --b 1'
args2='--pi gappy --beta 30'
args3="--hazard deterministic --chain 1"
args4='--datapoints 500 --datascale log --record 1 --recordutterances 1'
args="$args0 $args1 $args2 $args3 $args4"

# Parameters
gammas=(1 5 10 50	1000)
etas=(1 2 5 50 inf)
zetas=(1 1.5 2 5 inf)

# Skip existing experiments?
# If so, the script checks if a directory with the right name exists 
# and skips those. Note that it does not check the run. So if any
# run exists, the experiment will be skipped.
skipexisting=1

for gamma in "${gammas[@]}"
do
	for eta in "${etas[@]}"
	do
		for zeta in "${zetas[@]}"
		do
			
			# Setup source, output dir, params, name and go!
			src="../../src"
			out="../../results/BNG06/gamma-$gamma/eta-$eta/"
			params="--gamma $gamma --eta $eta --zeta $zeta"
			name="BNG06-gamma-$gamma-eta-$eta-zeta-$zeta"

			filecount=$(ls -l $out$name* 2>/dev/null | wc -l)
			if [[ $filecount -eq 0 || skipexisting -eq 0 ]] 
			then
				echo "Starting experiment $name"
				# File does not exist, so start the experiment
				mkdir -p $out
				python $src/BayesianNamingGame.py $args $params --name $name --out $out
			else
				# File already exsits, skipping
				echo ">>> Runs of $name already found; skipping..."
			fi

			echo "--------------------------------------------"
			echo ""
		done
	done
done