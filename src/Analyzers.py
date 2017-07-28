from helpers import *
import numpy as np

class Analyzer:
    def record(self, n, success, agents):
        return NotImplemented
    
    def export():
        return NotImplemented


class DetailedAnalyzer(Analyzer):
    """Collects detailed, per-word statistics"""
    
    def __init__(self, N, T, res, limit):
        self.res = res
        self.steps = T//res
        self.N = N
        self.limit = limit
        self.unique_counts = np.zeros((self.steps, limit), dtype=np.uint16)
        self.total_counts = np.zeros((self.steps, N, limit), dtype=np.uint16)
        self.successes = []
        
    def record(self, n, success, agents):
        if n % self.res != 0: return
            
        step = n//self.res
        
        # Count the total number of words per agent per object
        for a, agent in enumerate(agents):
            for i in range(1, self.limit+1):
                self.total_counts[step, a, i-1] = len(agent[i])

        # Count the number of unique words for every object
        for i in range(1, self.limit+1):
            words_for_i = set(flatten([a[i] for a in agents]))
            self.unique_counts[step, i-1] = len(words_for_i)
            
        self.successes.append(success)
        
    def export(self):
        return self.unique_counts, self.total_counts, self.successes
    

class PairAnalyzer(Analyzer):
    """Collects the unique pair count and total pair count"""
    
    def __init__(self, N, T, res, limit):
        self.res = res
        self.steps = T//res
        self.N = N

        self.unique_counts = np.zeros(self.steps, dtype=np.uint16)
        self.total_counts = np.zeros(self.steps, dtype=np.uint16)
        self.successes = np.zeros(self.steps)
        
    def record(self, n, success, agents):
        if n % self.res != 0: return
            
        step = n // self.res
        pairs = flatten([a.topairs() for a in agents])
        self.unique_counts[step] = len(set(pairs))
        self.total_counts[step] = len(pairs)
        self.successes[step] = success
        
    def export(self):
        return self.unique_counts, self.total_counts, self.successes
