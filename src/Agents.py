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


class MinimalNGAgent(list):
    """Implements the agent of a minimal naming game. Indeed, it is just a list"""
    def __init__(self, pop_voc):
        list.__init__(self)
        self.pop_voc = pop_voc
    
    def get_word(self):
        if len(self) == 0:
            self.append(self.pop_voc.generate_word())
        return choice(self)


class MultiWordMinimalNGAgent(defaultdict):
    """Implements the agent for a multi-word minimal naming game.
    That is basically a list of names for every object """
    
    def __init__(self, pop_voc):
        defaultdict.__init__(self, list)
        self.pop_voc = pop_voc
    
    def get_word(self, topic):
        if len(self[topic]) == 0:
            self[topic].append(self.pop_voc.generate_word())
        return choice(self[topic])
    
    def topairs(self):
        """Return a list of all (word, meaning) pairs"""
        return [(x, y) for x in self for y in self[x]]



class PopulationVocabulary:
    """Class that maintains a global vocabulary. This is only
    needed to ensure that new words are unique in the population"""
    
    def __init__(self):
        # The vocabulary contains numbers (they function as words)
        self.vocabulary = set()
        self.max_word = 0
        
        # The lexicon lexializes the numbers, just for readability
        self.lexicon = {}
    
    def generate_word(self):
        self.max_word += 1
        self.vocabulary.add(self.max_word)
        return self.max_word
    
    def lexicalize(self, word):
        
        if word not in self.lexicon.keys():
            cons, vowels = 'bcdfghjklmnpqrstvwxz', 'aeoui'
            lex = choice(cons) + choice(vowels) + choice(cons)
            while lex in self.lexicon:
                lex = choice(cons) + choice(vowels) + choice(cons)
        
            self.lexicon[word] = lex
        
        return self.lexicon[word]




class Lexicon(Counter):
    """Class for a simple lexicon with frequencies. The class just extends a 
    Counter with an update function."""
    
    def __init__(self, min_count=0):
        Counter.__init__(self)
        self.min_count = min_count
    
    def update_counts(self, x, update=1, update_others=0):
        """Update the frequencies of all items and update the frequencies 
        of the other items with another amount."""
        
        self[x] += update
        if update_others != 0:
            for other in list(self.keys()):
                if other == x: continue
                self[other] += update_others
                if self[other] <= self.min_count:
                    del self[other]



class PairLexicon(defaultdict):
    """A class for a lexicon with pairs (x,y) and their frequencies"""
    
    def __init__(self, min_count=0):
        """Pairs (x,y) are stored in a dictionary of counters of the form
        { x : Counter({y: 12}) }. This is a defaultdict, so you can always
        set/get entries directly, like `my_lexicon[x][y] = 10`"""
        defaultdict.__init__(self, Counter)
        self.min_count = min_count
    
    def update_counts(self, x, y, update=1, update_others=0):
        """Update the counts of (x,y) and also update the counts of
        all competing pairs (a,b) with either a=x or b=y."""
        orig_count = self[x][y]
        
        if update_others != 0:
            for a in list(self.keys()):
                for b in list(self[x].keys()):
                    if a == x or b == y:
                        self[a][b] += update_others
                        self.__clean_up(a, b)
        
        self[x][y] = orig_count + update
        self.__clean_up(x,y)
    
    def __clean_up(self, x, y):
        """Remove all entries with zero or negative count"""
        if self[x][y] <= self.min_count:
            del self[x][y]
            
        if len(self[x]) == 0:
            del self[x]

    def __repr__(self):
        return self.topairs(True).__repr__()
    
    def topairs(self, counts=False):
        """Return a list of all (word, meaning) pairs"""
        if counts:
            return dict(((x, y), f) for x in self 
                           for y, f in self[x].items())
        else:
            return [(x, y) for x in self for y in self[x]]
    
    def descendants(self, root, visited=[]):
        """All nodes accessible from the root (treating the pairs)
        as edges in a graph. Note that the list of nodes can contain
        duplicates."""
        if root not in self: return []
        if root in visited: return []
        
        children = list(self[root])
        visited.append(root)
        desc = [self.descendants(child, visited=visited) for child in children]
        return children + list(flatten(desc))
    
    def remove_inaccessible(self, root='START'):
        """Remove nodes that are not accessible from the root"""
        accessible = set(self.descendants(root, []))
        accessible.add(root)
        for word in list(self.keys()):
            if word not in accessible:
                del self[word]
