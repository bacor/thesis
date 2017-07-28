from random import choice
import numpy as np
from collections import defaultdict
from collections import Counter

class HurfordAdditiveAgent:
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


class HurfordMultiplicativeAgent:
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

