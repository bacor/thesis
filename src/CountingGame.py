import sys
sys.path.append('../')
from helpers import *
from collections import defaultdict
from collections import Counter


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
