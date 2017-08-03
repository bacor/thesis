from random import choice, sample
import numpy as np
from helpers import *
from collections import defaultdict, Counter
from AdditiveBaseGame import save_ABG_simulation, load_ABG_simulation
from scipy.sparse import dok_matrix

class MultiplicativeBaseGameAgent:
    """Agent that is also able to form multiplicative constructions."""

    def __init__(self, B=10, eta=1, simplify=True):
        self.freqs = Counter()
        self.simplify = simplify
        self.B = B
        self.eta = eta
        self.b0 = np.ceil((B+1)/2).astype(int)
        self.bases = np.arange(self.b0, B+1)
    
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
        """Find all favoured bases that could express n, or pick a random other 
        base if none of the favoured bases works
        
        How can we find all the expressions of the form $n = f \times b + r$?
        First, which bases $b$ can we use? Clearly, we cannot use $b$ if $n < b$. 
        Moreover, given the constraints $f \le b$ and $r \le B$, we can use base 
        $b$ to express numbers up to $b\times b + B$. In short: $n$ has a base-b
        representation if $b \le n \le b^2 + B$.
        
        Second, assuming $n$ is expressible, how do we find the decompositions 
        $n = f \times b + r$? Of course, $f = \text{floor}(\frac{n}{b})$ and 
        $r = n \% b$ is one solution, as in $26 = 4 \times 6 + 2$. But as the 
        remainder can be larger than the base, $26 = 3 \times 6 + 8$ is also a 
        valid expression. 
        
        In general, if $n = f \times b + r$, then $n = (f-1) \times b + (r + b)$ 
        is a valid expression if $r+b \le B$. Similar reasoning shows why in these 
        simulations a factor $f-2$ is always invalid: one would have to satisfy 
        $r + 2b \le B$, which is impossible if $b > \frac{1}{2} B$.
        """

        favoured = np.array(self.favoured_bases())
        idx = (favoured < n) * (n <= favoured**2 + self.B)
        bases = favoured[idx]

        # None of the favoured bases can express n
        if not idx.any():
            idx = (self.bases < n) * (n <= self.bases**2 + self.B)
            bases = self.bases[idx]
        
        # Decompose
        factors = n // bases
        remainders = n % bases

        # Expressions using factor f-1
        prev_idx = (remainders + bases <= self.B) * (factors >= 1)
        prev_bases = bases[prev_idx]
        prev_factors = factors[prev_idx] - 1
        prev_remainders = (remainders + bases)[prev_idx]
        
        # Concatenate everyting in a big matrix where every column
        # corresponds to a expressions (b, f, r). The matrix has
        # shape 3x[num expressions].
        expressions = np.array([
            np.concatenate([bases, prev_bases]), 
            np.concatenate([factors, prev_factors]),
            np.concatenate([remainders, prev_remainders])])

        # Ensure that factor <= base
        okay = expressions[1,:] <= expressions[0,:]
        expressions = expressions[:, okay]
        
        # If there are any simple expression (r=0), restrict to those
        simplest_idx = expressions[2,:] == 0
        if np.any(simplest_idx) and self.simplify:
            expressions = expressions[:,simplest_idx]
                    
        # Randomly pick an expression
        i = np.random.randint(expressions.shape[1])
        return expressions[:,i]


def MultiplicativeBGSimulation(T=5000, N=200, res=10, B=10, n_min=None, n_max=None, **kwargs):
    """Run Hurfords experiment. With a certain resolution we compute for every 
    agent which bases it favours. This is stored in one-hot format. So if there
    are 5 potential bases (6,7,8,9,10), N agents and T timesteps, you get an
    (T/res) x N x 5 array.
    
    T: Timesteps
    N: Number of agents
    """
    agents = [MultiplicativeBaseGameAgent(B=B, **kwargs) for _ in range(N)]
    favoured, successes = [], []
    b0 = np.ceil((B+1)/2).astype(int)
    n_min = B+1 if n_min is None else n_min
    n_max = B*B + B if n_max is None else n_max
    _num_poss_bases = len(range(b0, B+1))

    # Quantities to track
    D = int(np.ceil(T / res)) # num datapoints
    num_bases = np.zeros(D)
    num_unique_bases = np.zeros(D)
    base_counts = dok_matrix((D, _num_poss_bases), dtype=int)
    successes = np.zeros(D)
    
    for t in range(T):
        s, h = np.random.randint(0, high=len(agents), size=2)
        speaker, hearer = agents[s], agents[h]
        n = np.random.randint(n_min, n_max+1)
        
        expr = speaker.express(n)
        base = max(expr[:-1])
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


def save_MBG_simulation(params, results, directory, name):
    return save_ABG_simulation(params, results, directory, name)
    
def load_MBG_simulation(directory, name, params_only=False):
    return load_ABG_simulation(directory, name, params_only)


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
    
    # Optional
    parser.add_argument('--nmin', type=int, default=None)
    parser.add_argument('--nmax', type=int, default=None)
    parser.add_argument('--simplify', type=int, default=1)
    args = parser.parse_args()

    if os.path.isdir(args.out) == False:
        raise NotADirectoryError('The output directory could not be found.')
    
    params = dict(
        N=args.agents,
        T=args.timesteps,
        B=args.B,
        res=args.res,
        eta=args.eta,
        n_min=args.nmin,
        n_max=args.nmax,
        simplify=args.simplify==1)

    results = repeat_simulation(MultiplicativeBGSimulation, args.runs, **params)
    
    params['name'] = args.name
    params['runs'] = args.runs
    save_MBG_simulation(params, results, args.out, args.name)
