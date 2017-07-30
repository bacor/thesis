# Experiment LING01
# 
# Lateral inhibition name for various values of d_inh

args='--agents 200 --timesteps 30000 --limit 1 --res 100 --runs 300'

src='../../src'
out='results/'

# Strategies
minimal='--dinit 1 --dinc 0 --dinh 1 --ddec 0 --smax 0'
frequency='--dinit 1 --dinc 1 --dinh 0 --ddec 0 --smax 0'
li1='--dinit 1 --dinc 1 --dinh 1 --ddec 0 --smax 0'
li2='--dinit .5 --dinc .1 --dinh .5 --ddec .1 --smax 1'
li3='--dinit .5 --dinc .1 --dinh .2 --ddec 0.2 --smax 0'


Minimal strategy
echo "Starting experiment with minimal strategy"
python $src/LateralInhibitionNamingGame.py $minimal $args --name LING01-minimal --out $out

# Frequency strategy
echo "Starting experiment with frequency strategy"
python $src/LateralInhibitionNamingGame.py $frequency $args --name LING01-frequency --out $out

# LI strategy 1
echo "Starting experiment with LI strategy 1"
python $src/LateralInhibitionNamingGame.py $li1 $args --name LING01-li1 --out $out

# LI strategy 2
echo "Starting experiment with LI strategy 2"
python $src/LateralInhibitionNamingGame.py $li2 $args --name LING01-li2 --out $out

# LI strategy 3
echo "Starting experiment with LI strategy 3"
python $src/LateralInhibitionNamingGame.py $li3 $args --name LING01-li3 --out $out

echo "Done"
