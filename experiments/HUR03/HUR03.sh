# Experiment HUR02
# 
# In this experiment we run Hurfords second basic simulation 
# (with additive agents) for various values of B
# 

args='--eta 1 --B 10 --res 10 --timesteps 8000 --runs 600 --agents 200'

#python HUR01.py $args --B 10 --name HUR03-additive
python ../../src/MultiplicativeBaseGame.py $args --B 10 --name HUR03-multiplicative

