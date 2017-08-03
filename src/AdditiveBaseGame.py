from random import choice, sample
import numpy as np
from helpers import *
from collections import defaultdict
from collections import Counter
from scipy.sparse import dok_matrix
import os
import json

class AdditiveBaseGameAgent:
    """Agent that can only formulate additive constructions"""

    def __init__(self, B=10, eta=1):
        self.freqs = Counter()
        
        self.eta = eta
        self.b0 = np.ceil((B+1)/2).astype(int)
        self.bases = list(range(self.b0, B+1))
    
    def favoured_bases(self, **kwargs):
        # Eta is the factor used in Hurfords other criterion,
        # Using eta=1 (default) corresponds to the max-freq criterion
        eta = self.eta if 'eta' not in kwargs else kwargs['eta']
        if len(self.freqs) > 0:
            maximum = max(self.freqs.values())
        else:
            maximum = 0
        return [b for b in self.bases 
                if b in self.freqs and self.freqs[b] >= maximum/eta]

    def express(self, n):
        # Find all favoured bases that could express n, or pick
        # a random other base if none of the favoured bases works
        candidates = [b for b in self.favoured_bases() if n <= b+b]
        if len(candidates) == 0:
            candidates = [b for b in self.bases if n <= b+b]
        
        base = choice(candidates)
        return [base, n - base]

def AdditiveBGSimulation(T=5000, N=200, res=10, B=10,
    init_base=10, init_freq=1, init_frac=0, **kwargs):
    """Run Hurfords experiment. With a certain resolution we compute for every 
    agent which bases it favours. This is stored in one-hot format. So if there
    are 5 potential bases (6,7,8,9,10), N agents and T timesteps, you get an
    (T/res) x N x 5 array.
    
    T: Timesteps
    N: Number of agents
    """
    agents = [AdditiveBaseGameAgent(B=B, **kwargs) for _ in range(N)]
    b0 = np.ceil((B+1)/2).astype(int)
    _num_poss_bases = len(range(b0, B+1))

    # Bias
    if init_frac > 0:
        num_biased_agents = round(init_frac * N)
        for agent in agents[:num_biased_agents]:
            agent.freqs[init_base] = init_freq
    
    # Quantities to track
    D = int(np.ceil(T / res)) # num datapoints
    num_bases = np.zeros(D)
    num_unique_bases = np.zeros(D)
    base_counts = dok_matrix((D, _num_poss_bases), dtype=int)
    successes = np.zeros(D)
    
    for t in range(T):
        s, h = sample(range(N), 2)
        speaker, hearer = agents[s], agents[h]
        n = np.random.randint(B+1, 2*B+1)
        
        expr = speaker.express(n)
        base = max(expr)
        success = base in hearer.favoured_bases()
        hearer.freqs[base] += 1
        
        if t % res == 0:
            idx = t//res
            
            # Get one-hot representation of the favoured bases of all agents
            fav_bases = []
            for a in agents:
                fav = a.favoured_bases()
                onehot = np.array([b in fav for b in range(b0, B+1)])
                fav_bases.append(onehot)
            fav_bases = np.array(fav_bases)
            
            # Store relevant quantities
            base_counts[idx,:] = fav_bases.sum(axis=0)
            num_bases[idx] = fav_bases.sum()
            num_unique_bases[idx] = (fav_bases.sum(axis=0) > 0).sum()
            successes[idx] = success

    return base_counts, num_bases, num_unique_bases, successes

def save_ABG_simulation(params, results, directory, name):
    base_counts, num_bases, num_unique_bases, successes = results
    _base = os.path.join(directory, name)
    
    base_counts = np.array([M.todense() for M in base_counts])
    np.save(_base+'-base-counts.npy', base_counts, allow_pickle=False)
    np.savetxt(_base+'-num-bases.txt.gz', np.array(num_bases))
    np.savetxt(_base+'-num-unique-bases.txt.gz', np.array(num_unique_bases))
    np.savetxt(_base+'-successes.txt.gz', np.array(successes))
    json.dump(params, open(_base+'-params.json', 'w'))
    
def load_ABG_simulation(directory, name, params_only=False):
    _base = os.path.join(directory, name)
    params = json.load(open(_base+'-params.json', 'r'))
    b0 = np.ceil((params['B']+1)/2).astype(int)
    params['bases'] = range(b0, params['B']+1)
    if params_only: return params
    
    base_counts = np.load(_base+'-base-counts.npy', allow_pickle=False)
    num_bases = np.loadtxt(_base+'-num-bases.txt.gz')
    num_unique_bases = np.loadtxt(_base+'-num-unique-bases.txt.gz')
    successes = np.loadtxt(_base+'-successes.txt.gz')
    return base_counts, num_bases, num_unique_bases, successes, params


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
    parser.add_argument('--initfreq', type=float)
    parser.add_argument('--initbase', type=int)
    
    # Optional
    parser.add_argument('--initfrac', type=float, default=1)
    
    args = parser.parse_args()
    if os.path.isdir(args.out) == False:
        raise NotADirectoryError('The output directory could not be found.')
    
    params = dict(
        N=args.agents,
        T=args.timesteps,
        B=args.B,
        res=args.res,
        eta=args.eta,
        init_frac=args.initfrac,
        init_base=args.initbase,
        init_freq=args.initfreq)

    results = repeat_simulation(AdditiveBGSimulation, args.runs, **params)
    
    params['name'] = args.name
    params['runs'] = args.runs
    save_ABG_simulation(params, results, args.out, args.name)