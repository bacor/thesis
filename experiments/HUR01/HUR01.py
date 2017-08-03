from collections import Counter
from random import choice
import numpy as np
import sys

# Local dependencies
sys.path.append('../')
from helpers import *
from Agents import HurfordAdditiveAgent


def HurfordSimulation01(T=5000, N=200, res=10, B=10, **kwargs):
    """Run Hurfords experiment. With a certain resolution we compute for every 
    agent which bases it favours. This is stored in one-hot format. So if there
    are 5 potential bases (6,7,8,9,10), N agents and T timesteps, you get an
    (T/res) x N x 5 array.
    
    T: Timesteps
    N: Number of agents
    """
    agents = [HurfordAdditiveAgent(B=B, **kwargs) for _ in range(N)]
    favoured, successes = [], []
    b0 = np.ceil((B+1)/2).astype(int)
    
    for i in range(T):
        s, h = np.random.randint(0, high=len(agents), size=2)
        speaker, hearer = agents[s], agents[h]
        n = np.random.randint(B+1, 2*B+1)
        
        expr = speaker.express(n)
        base = max(expr)
        success = base in hearer.favoured_bases()
        hearer.freqs[base] += 1

        if i % res == 0:
            
            # Record whether communication was successful
            successes.append(success)
            
            # Get one-hot representation of the favoured bases of all agents
            fav_bases = []
            for a in agents:
                fav = a.favoured_bases()
                onehot = np.array([b in fav for b in range(b0, B+1)])
                fav_bases.append(onehot)
            favoured.append(fav_bases)
        
    return np.array(favoured), np.array(successes)


##########################

if __name__ == '__main__':
    import argparse
    import os
    import pickle
    import json

    # Define all command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--res',  type=int, required=True)
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--agents', type=int, required=True)
    parser.add_argument('--B', type=int, required=True)
    parser.add_argument('--eta',  type=float, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()

    if os.path.isdir('results') == False:
        raise NotADirectoryError('The output directory could not be found.')
    
    setup = dict(
        N=args.agents,
        T=args.timesteps,
        B=args.B,
        res=args.res,
        eta=args.eta)

    results = repeat_simulation(HurfordSimulation01, args.runs, **setup)
    
    # Store all results directly.
    fn = '{}.pickle'.format(args.name)
    pickle.dump(results, open(os.path.join(args.out, fn), 'wb'))

    # Dump the setup to a JSON file
    setup['name'] = args.name
    setup['runs'] = args.runs
    logfn = '{}-setup.json'.format(args.name)
    json.dump(setup, open(os.path.join(args.out, logfn), 'w'))

