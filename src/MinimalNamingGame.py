import numpy as np
import sys
sys.path.append('../../src')
from helpers import *
from random import choice, sample
import numpy as np
from collections import defaultdict, Counter
import argparse
import os
import pickle
import json


class MinimalNGAgent(list):
    """Implements the agent of a minimal naming game. Indeed, it is just a list"""
    def __init__(self, pop_voc):
        list.__init__(self)
        self.pop_voc = pop_voc
    
    def get_word(self):
        if len(self) == 0:
            self.append(self.pop_voc.generate_word())
        return choice(self)


def MinimalNGSimulation(N=200, T=10000, res=20):
    pop_voc = PopulationVocabulary()
    agents = [MinimalNGAgent(pop_voc) for i in range(N)]
    num_unique_words, num_words, successes = [], [], []
    
    for n in range(T):
        s, h = sample(range(N), 2)
        speaker, hearer = agents[s], agents[h]

        word = speaker.get_word()
        success = word in hearer
        if word in hearer:
            speaker[:] = [word]
            hearer[:] = [word]
        else: 
            hearer.append(word)

        if n%res == 0:
            num_unique = len(set(flatten([a for a in agents ])))
            total = sum(list(map(len, agents)))
            num_unique_words.append(num_unique)
            num_words.append(total)
            successes.append(success)
            
    return num_unique_words, num_words, successes, agents


def load_MNG_simulation(directory, name, load_agents=False, params_only=False):
    """
    Load Minimal Naming Game from files
    """
    res = {}
    
    fn = os.path.join(directory, f'{name}-params.json')
    res['params'] = json.load(open(fn, 'r'))
    if params_only: return res['params']
    
    fn = os.path.join(directory, f'{name}-num-unique-words.csv.gz')
    res['num_unique_words'] = np.loadtxt(fn)
    
    fn = os.path.join(directory, f'{name}-num-words.csv.gz')
    res['num_words'] = np.loadtxt(fn)
    
    fn = os.path.join(directory, f'{name}-successes.csv.gz')
    res['successes'] = np.loadtxt(fn)
    
    if load_agents:
        fn = os.path.join(directory, f'{name}-agents.pickle')
        res['agents'] = pickle.load(open(fn, 'rb'))
    
    return res


def save_MNG_simulation(directory, name, setup, results):
    num_unique_words, num_words, successes, agents = results

    fn = os.path.join(directory, f'{name}-num-unique-words.csv.gz')
    np.savetxt(fn, np.array(num_unique_words))

    fn = os.path.join(directory, f'{name}-num-words.csv.gz')
    np.savetxt(fn, np.array(num_words))

    fn = os.path.join(directory, f'{name}-successes.csv.gz')
    np.savetxt(fn, np.array(successes))

    fn = os.path.join(directory, f'{name}-agents.pickle')
    pickle.dump(agents, open(fn, 'wb'))

    fn = os.path.join(directory, f'{name}-params.json')
    json.dump(setup, open(fn, 'w'))


##########################

if __name__ == '__main__':

    # Define all command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--res',  type=int, required=True)
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--agents', type=int, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()

    if os.path.isdir(args.out) == False:
        raise NotADirectoryError('The output directory could not be found.')
    
    setup = dict(
        N=args.agents,
        T=args.timesteps,
        res=args.res)

    # Run!
    results = repeat_simulation(MinimalNGSimulation, args.runs, **setup)
    
    # Store all results directly.
    setup['name'] = args.name
    setup['runs'] = args.runs
    save_MNG_simulation(args.out, args.name, setup, results)
