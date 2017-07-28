from random import choice
import numpy as np
from scipy.sparse import csr_matrix
from itertools import chain

def choiceDict(dictionary):
    """Randomly pick key, value from dictionary"""
    key = choice(list(dictionary.keys()))
    return key, dictionary[key]

def flatten(mylist):
    """Flatten a list"""
    return list(chain.from_iterable(mylist))

def repeat_simulation(simulate, runs, **kwargs):
    """Runs the function `simulate` several times, using the arguments
    passed as keywords. The output has the same length as the output of
    `simulate`, but each of the outputs is now a list of length `runs`.
    So if `simulate` returns (bases, favs) and you do 2 runs, you get back
    ([bases1, bases2], [favs1, favs2]).
    """
    
    # Do first run to determine number of outputs
    first_run = simulate(**kwargs)
    results = [[out] for out in first_run]
    
    # Do the rest of the simulations
    for run in range(runs-1):
        outputs = simulate(**kwargs)
        for i, out in enumerate(outputs):
            results[i].append(out)
            
    return results

def save_csr(data, fn):
    """Exports a CSR sparse matrix to three gzipped text files"""
    fn += '-{}.txt.gz'
    np.savetxt(fn.format('indices'), data.indices, fmt='%i')
    np.savetxt(fn.format('data'), data.data, fmt='%i')
    np.savetxt(fn.format('indptr'), data.indptr, fmt='%i')

def load_csr(basename, shape):
    """Load a CSR sparse matrix saved in three gzipped files"""
    fn = basename + '-{}.txt.gz'
    data = np.loadtxt(fn.format('data'))
    indices = np.loadtxt(fn.format('indices'))
    indptr = np.loadtxt(fn.format('indptr'))
    return csr_matrix((data, indices, indptr), shape=shape)

def entropy(ps):
    """Compute the base-2 Shannon entropy of distributions in ps
    Every row should correspond to a distribution."""
    single_dist = len(ps.shape) == 1
    if single_dist: ps = np.array([ps]);
    # H = -1 * np.sum(ps * np.log2(ps), axis=1)
    H = -1 * np.sum(np.multiply(ps, np.ma.log2(ps).filled(0)), axis=1)
    return H[0] if single_dist else H
    
def JSD(ps):
    """
    Computes the Jensen-Shannon divergence of distributions p_1, ..., p_N:

        JSD(p_1, ..., p_N) = H( mean(p_1, ... p_N) ) - mean( H(p_1), ..., H(p_N) )
    
    where H is the (base-2) Shannon-entropy H(X) = E[ -log2(X) ].

    Args:  
        ps: a matrix with one distribution per row.
    """
    term1 = entropy(ps.mean(axis=0))
    term2 = np.mean(entropy(ps))
    return term1 - term2

def norm_JSD(ps, K=None):
    K = K if K else ps.shape[1]
    return JSD(ps) / np.log2(K)

def join(*arrs, cols=None):
    """Joins arrays into a 2D matrix with K columns. If the array doesn't
    have K columns already, it is transposed if it does have K rows.
    
    Args:
        a list of arrays
        cols: the number of columsn (required)

    Returns:
        A ?-by-cols matrix 
    """
    if not cols:
        raise ValueError('Number of colums not specified')
        
    to_join = []
    for arr in arrs:
        if len(arr.shape) == 1:
            if arr.shape[0] == cols:
                to_join.append([arr])
            else:
                raise ValueError('Array does not have appropriate size')
        else:
            if arr.shape[1] == cols:
                to_join.append(arr)
            elif arr.shape[0] == cols:
                to_join.append(arr.T)
            else:
                raise ValueError('Array does not have appropriate size')
    return np.concatenate(to_join)

def normalize(arr, axis=1):
    """Normalize an array along a certain axis"""
    _arr = arr if type(arr) == np.ndarray else np.array(arr)
    if len(_arr.shape) == 1:
        return normalize([arr])[0]
    else:
        return _arr / _arr.sum(axis=axis)[:, np.newaxis]


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

