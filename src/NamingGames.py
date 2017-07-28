from random import choice, sample
import numpy as np
from collections import defaultdict, Counter
from helpers import *

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


class LIAgent(defaultdict):
    """A class for a lexicon with pairs (x,y) and their frequencies"""
    
    def __init__(self, lexicon, d_inc=1, d_inh=1, d_init=1, d_dec=0, 
                 min_score=0, max_score=10e6):
        """Pairs (x,y) are stored in a dictionary of counters of the form
        { x : Counter({y: 12}) }. This is a defaultdict, so you can always
        set/get entries directly, like `my_lexicon[x][y] = 10`
        
        Args:
            - d_inc: on success, scores of pair are increased by d_inc (default 1)
            - d_inh: on a success, scores of competing pairs are reduced by d_inh (default 1)
            - d_init: on failure, hearers score of pair is increased to d_init (default 1)
            - d_dec: on failure: speakers score of pair si decreased by d_dec (default 0)
            - min_score: minimum allowed score (default 0)
            - max_score: maximum score allowed (default 10e6)
        """
        defaultdict.__init__(self, Counter)
        self.lexicon = lexicon
        
        self.d_inh = d_inh
        self.d_init = d_init
        self.d_dec = d_dec
        self.d_inc = d_inc
        self.min_score = min_score
        self.max_score = max_score
    
    def update(self, x, y, is_success, is_speaker):
        if is_success:
            self.update_score(x, y, increase=self.d_inc, inhibit=self.d_inh)
            
        else:
            if is_speaker:
                self.update_score(x, y, increase=-1*self.d_dec, inhibit=0)
            else:
                self.update_score(x, y, increase=self.d_init, inhibit=0)
    
    def update_score(self, x, y, increase, inhibit):
        """Update the scores of (x,y) and also update the scores of
        all competing pairs (a,b) with either a=x or b=y."""
        orig_score = self[x][y]
        
        if inhibit != 0:
            for a in list(self.keys()):
                for b in list(self[a].keys()):
                    if a == x or b == y:
                        self[a][b] -= inhibit
                        self.__clean_up(a, b)
        
        self[x][y] = min(orig_score + increase, self.max_score)
        self.__clean_up(x,y)
    
    def __clean_up(self, x, y):
        """Remove all entries with zero or negative count"""
        if self[x][y] <= self.min_score:
            del self[x][y]
            
        if len(self[x]) == 0:
            del self[x]

    def __repr__(self):
        return self.topairs(True).__repr__()
    
    def get_y(self, x):
        if len(self[x]) == 0:
            y = self.lexicon.generate_word()
            self[x][y] = self.d_init
            return y
        
        else:
            # I profiled this, seems fine.
            _, max_score = self[x].most_common(1)[0]
            candidates = [y for y, f in self[x].items() if f == max_score]
            return choice(candidates)
    
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
                
    
def LISimulate(Agent, Analyzer, N=200, T=10000, res=100, limit=2, **kwargs):
    
    # Initialize population
    lexicon = PopulationVocabulary()
    agents = [Agent(lexicon, **kwargs) for _ in range(N)]
    
    # Initialize analyzer
    analyzer = Analyzer(N=N, T=T, res=res, limit=limit)
    
    for n in range(T):
        s, h = sample(range(N), 2)
        speaker, hearer = agents[s], agents[h]
        
        # Get pair (x, y)
        x = np.random.randint(1, limit+1)
        y = speaker.get_y(x)
        success = y in hearer[x]
        
        # Update speaker/hearer
        hearer.update(x, y, success, False)
        speaker.update(x, y, success, True)

        # Record statistics
        analyzer.record(n, success, agents)
            
    return analyzer.export()


def testLIAgent():
    """Some tests for LIAgent"""

    # Tests for the minimal naming game
    if True:
        lexicon = PopulationVocabulary()
        A = LIAgent(lexicon, d_init=1, d_inh=1, d_dec=0, d_inc=1, max_score=1)

        # Successful communication of (1, een) by speaker/hearer
        # Repeated successes should not change the frequency
        for i in range(10):
            is_speaker = i%2 == 0
            A.update(1, 'een', True, is_speaker)
            assert A[1]['een'] == 1 
            assert len(A) == 1
            assert len(A[1]) == 1

        # Successfull/failed communication of (1, 'one') by speaker;
        # behaviour should be identical
        for i in range(10):
            is_success = i%2 == 0
            A.update(1, 'one', is_success, True)
            assert A[1]['one'] == 1
            assert len(A[1]) == 1

        # Succesfully communicate (i, 'i'); i = 1, ..., 10
        for i in range(1, 11):
            A.update(i, str(i), True, True)
        assert len(A) == 10

        # Reset
        for i in range(1, 11):
            A.update_score(i, str(i), -1, 0)
        assert len(A) == 0

        # Failed communication for hearer
        for i in range(1,11):
            A.update(1, str(i), False, False)
            assert len(A[1]) == i
            assert A[1][str(i)] == 1

        # See if inhibition works properly
        A.update(1, '5', True, False)
        assert len(A[1]) == 1
        assert A[1]['5'] == 1


        # See if the generation works fine
        assert A.get_y(1) == '5'

        for i in range(10,20):
            y = A.get_y(i)
            y = A.get_y(i)
            y = A.get_y(i)
            assert y == lexicon.max_word

        # See if you indeed roughly sample uniformly from the two most frequent things
        A.update(1, '6', False, False)
        sample = [A.get_y(1) for _ in range(10000)]
        assert abs(5000 - Counter(sample)['5']) < 200

    # Test for frequency strategy
    if True:
        lexicon = PopulationVocabulary()
        A = LIAgent(lexicon, d_init=1, d_inh=0, d_dec=0, d_inc=1)

        # Observe 10 (1, 1) pairs
        for i in range(1, 11):
            A.update(1, 1, True, True)
            A.update(1, 2, True, True)

        assert len(A[1]) == 2
        assert  A[1][1] == 10
        assert  A[1][2] == 10

        # Failed update for hearer
        A.update(2, 2, False, True)
        assert len(A) == 1

        # Failed update for hearer
        A.update(2, 2, False, False)
        A.update(2, 2, False, False)
        assert len(A) == 2
        assert A[2][2] == 2

        # See if you indeed roughly sample uniformly from the two most frequent things
        sample = [A.get_y(1) for _ in range(10000)]
        assert abs(5000 - Counter(sample)[1]) < 200

        # Newly generated words?
        A.get_y(3)
        assert A.get_y(3) == 1
        assert len(A[3]) == 1

    # Test for lateral inhibition strategy 1
    if True:
        lexicon = PopulationVocabulary()
        A = LIAgent(lexicon, d_init=1, d_inh=1, d_dec=0, d_inc=1)

        # Speaker, succesfull communication (twice)
        A.update(1, 'een', True, True)
        A.update(1, 'een', True, True)
        assert len(A) == 1
        assert len(A[1]) == 1
        assert A[1]['een'] == 2

        # Speaker, succesfull comm using diff word for same obj
        A.update(1, 'one', True, True)
        assert len(A[1]) == 2
        assert A[1]['een'] == 1 # Inhibited!
        assert A[1]['one'] == 1

        # Speaker, diff object
        for i in range(4):
            A.update(2, 'twee', True, True)
        assert len(A) == 2
        assert A[2]['twee'] == 4

        # Use same word elsewhere
        A.update(3, 'twee', True, True)
        assert A[3]['twee'] == 1
        assert A[2]['twee'] == 3 # Inhibited!

        # Failed update speaker, nothing hapens
        A.update(4, 'vier', False, True)
        assert len(A) == 3

        # Faile updat hearer
        A.update(4, 'vier', False, False)
        assert len(A) == 4
        assert A[4]['vier'] == 1

        # Randomly choose between most frequent ones
        sample = [A.get_y(1) for _ in range(10000)]
        assert abs(5000 - Counter(sample)['een']) < 200

        # Newly generated words
        assert A.get_y(5) == 1
        assert len(A[5]) == 1
