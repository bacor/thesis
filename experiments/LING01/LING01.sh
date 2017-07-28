# Experiment LING01
# 
# Lateral inhibition name for various values of d_inh

args='--agents 200 --timesteps 10000 --limit 1 --res 100 --runs 300'


src='../../src'
out='results/'

# Strategies
minimal='--dinit 1 --dinc 0 --dinh 1 --ddec 0'
frequency='--dinit 1 --dinc 1 --dinh 0 --ddec 0'
simpleli='--dinit 1 --dinc 1 --dinh 1 --ddec 0'
li='--dinit .5 --dinc .1 --dinh .2 --ddec .2'


# Minimal strategy
echo "Starting experiment with minimal strategy"
python $src/LateralInhibitionNamingGame.py $minimal $args --name LING01-minimal --out $out

# Frequency strategy
echo "Starting experiment with frequency strategy"
python $src/LateralInhibitionNamingGame.py $frequency $args --name LING01-frequency --out $out

# Simple LI strategy
echo "Starting experiment with simple li strategy"
python $src/LateralInhibitionNamingGame.py $simpleli $args --name LING01-simple-li --out $out

# LI strategy
echo "Starting experiment with LI strategy"
python $src/LateralInhibitionNamingGame.py $li $args --name LING01-li --out $out

echo "Done"
