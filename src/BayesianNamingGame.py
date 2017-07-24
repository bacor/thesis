from Weibull import *
from helpers import *
import numpy as np
from random import sample
import pandas as pd
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from scipy.stats import linregress
from scipy.misc import logsumexp
import json
import os
import time

def BNG_simulation(
    T=5, K=50, N=10, b=1,
    eta=1,   # Lang-sampling strategy
    zeta=1,  # Word-sampling strategy
    gamma='inf', # Life expectancy    
    alpha=None, init_counts=None,
    chain=True, hazard='weibull',
    num_datapoints=500, datascale='log',
    record=True,
    record_utterances=True):
    """
    Simulate the Bayesian Naming Game.
    
    Params
        - T: Number of iterations
        - K: Number of words (categories)
        - N: Number of agents
        - b: bottleneck; number of communicated words
        - eta: strategy parameter for selecting languages. Interpolates between
            samplers (eta=1) and MAP (zeta='inf')
        - zeta: strategy parameter for selecting words. Interpolates between 
            samplers (zeta=1) and MAP (zeta='inf'). 
        - gamma: life expectancy parameter. Interpolates between iterated
            learing (gamma=1) and the naming game (gamma = 'inf')
        - alpha: hyperparameter for the Dirichlet at time 0, shared by all agents.
            Should be a K-vector 
        - counts: initial counts for each of the agents. NxK matrix
        - chain: place the agents in a chain so that the previous hearer
            is the next speaker. If false, agents are randomly sampled in
            every round.
        - hazard: the hazard function to use
            "weibull": a discrete Weibull hazard with mean gamma > 1
            "geometric": a geometric (constant) hazard function with 
                parameter 1/gamma; gamma > 1
            "deterministic": agents die after gamma interactions
        - num_datapoints: the number of datapoints to collect
        - datascale: collect datapoints on a linear or logarithmic scale?
            can be either "linear" or "log" (default)
            
    Returns: 
        a dictionary with the following fields:
        - N, T, b, alpha, gamma, init_counts, zeta: all the parameter values
        - utterances: a TxK sparse matrix in CSR format with the utterance uttered
            in every round. The t'th row thus contains for each of the K word, how
            many times it was uttered in round t. Rows sum up to b.
        - stats: statistics that are periodically collected (at the end of the
            round.) This is a Tx3 matrix where columsn contain 
                0. JSD(phi_1, ..., phi_N); 
                1. JSD(alpha, phi_1, ..., phi_N)
                2. JSD(alpha, mean(phi_1, ... phi_N))
        - recording: a pandas dataframe with data describing every round. More
            precisely, it lists the speaker, hearer, age of both and whether the
            speaker died at the end of that round
    """

    # Approximate 'infinity'
    inf = 1e20
    gamma = max(inf, 10*T) if gamma == 'inf' else float(gamma)
    eta = inf if eta == 'inf' else float(eta)
    zeta = inf if zeta == 'inf' else float(zeta)

    # Check parameters
    assert T >= 1
    assert K >= 1
    assert N >= 1
    assert b >= 1
    assert gamma >= 1
    assert zeta >= 0
    assert eta >= 0
    
    # Hazard function
    if hazard == 'weibull':
        W = SingleParamDiscreteWeibull(gamma)
        hazard_fn = lambda t: W.hazard(t)
    elif hazard == 'geometric':
        hazard_fn = lambda t: t*0 + 1.0/gamma
    elif hazard == 'deterministic':
        hazard_fn = lambda t: t >= gamma
    else:
        raise ValueError('hazard should be one of "weibull", "geometric", "deterministic"')
    
    # Some trivial tests
    for t in np.arange(0, min(T, 1000000), 100):
        h = hazard_fn(t)
        assert h >= 0 and h <= 1
    
    # Check shape of alpha.
    alpha_ps = alpha / alpha.sum()
    assert len(alpha) == K

    # Initialize counts
    counts = np.array(init_counts) if not type(init_counts) == type(None) else np.zeros((N,K))
    assert counts.shape == (N,K)
    
    # Track the ages of the speakers (num interactions)
    age = np.zeros(N)
    
    if record:
        # Track the exact game
        recording = np.zeros((T, 5), dtype=int)
    
    if record_utterances:
        # Store the words uttered in a sparse matrix.
        # In the end, it is converted to a CSR sparse matrix
        utterances = dok_matrix((T, K), dtype=int)
    
    # Collect statistics
    if datascale == 'log':
        datapoints = np.unique([int(10**t) for t in 
                        np.linspace(0, np.log10(T)+1, num_datapoints)])
        datapoints = datapoints[datapoints <= T] 
    elif datascale == 'linear':
        datapoints = np.arange(0, T, num_datapoints)
    assert datapoints[-1] <= T
    statistics = np.zeros((len(datapoints), 4))
    idx = 0
    
    # Pick the very first agent in the chain (unused if chain==False)
    prev_hearer, = sample(range(N), 1)
    
    # Go!
    for t in range(T):
        
        # Pick the speaker and hearer
        if chain:
            s = prev_hearer
            h, = sample([j for j in range(N) if j != s], 1)
            prev_hearer = h
        else:
            s, h = sample(range(N), 2)
        assert s != h
        
        #############################################################
        # This is where it all happens...
        # 
        
        # The speaker draws a language, biased towards the mode by eta
        _alpha = eta*(alpha + counts[s,:] - 1) + 1 
        _theta = np.random.dirichlet(_alpha)
        
        # Then exaggerate the language by zeta
        _logtheta = zeta * np.log(_theta)
        theta = np.exp(_logtheta - logsumexp(_logtheta))
        
        # The speaker draws b utterances from theta 
        utterance = np.random.multinomial(b, pvals=theta)

        # Update the counts of hearer. This is the update of the posterior 
        # of the hearer; in the next round it will use an updated alpha.
        counts[h, :] += utterance
        
        #############################################################
        
        # Update speaker age: number of interactions
        age[s] += 1
        
        # Decide if the speaker dies
        speaker_dies = np.random.rand() < hazard_fn(age[s])
        
        # Store all data
        if record:
            recording[t,:] = (s, h, age[s], age[h], speaker_dies)

        if record_utterances:
            utterances[t,:] = utterance
        
        # Reset dead speakers
        if speaker_dies:
            counts[s,:] = 0 # Amounts to alpha^(t) := alpha^(0)
            age[s] = 0
            
        # Collect general statistics after certain intervals
        if idx < len(datapoints) and t == datapoints[idx]:
            
            # Predictive distribution for a single (!) word (b=1)
            vs = alpha + counts
            phis = vs / vs.sum(axis=1)[:,np.newaxis]
            mphi = phis.mean(axis=0)
            
            # Compute JSD between all predictive distributions (phis) and 
            # also to the prior predictive (alpha_ps).
            jsd_phis = JSD(phis)
            jsd_alpha_phis = JSD(np.concatenate(([alpha_ps], phis), axis=0))
            jsd_alpha_mphi = JSD(np.concatenate(([alpha_ps], [mphi]), axis=0))
            entropy_mphi = entropy(mphi)
            statistics[idx,:] = (jsd_phis, jsd_alpha_mphi, entropy_mphi, jsd_alpha_phis)
            idx += 1

    # Export data to Pandas dataframe and return
    if record:
        columns = ['speaker', 'hearer', 'speaker_age', 'hearer_age', 'death']
        rec_df = pd.DataFrame(recording, index=np.arange(T), columns=columns)
        rec_df['death'] = rec_df['death'] == 1
        rec_df.index.title = 'time'
    else: 
        rec_df = None

    if record_utterances:
        utter = utterances.tocsr()
    else:
        utter = None 
    
    columns = ['JSD(*phis)', 'JSD(alpha, mean(*phis))', 'H(mean(*phi))', 'JSD(alpha, *phis)',]
    stats_df = pd.DataFrame(statistics, index=datapoints, columns=columns)
    stats_df.index.title = 'time'

    # Final distribution
    vs = alpha + counts
    phis = vs / vs.sum(axis=1)[:,np.newaxis]

    return {
        'alpha': alpha.tolist(), 
        'beta': alpha.sum(),
        'pi': alpha_ps.tolist(),
        'init_counts': np.array(init_counts).tolist() if not type(init_counts) == type(None) else None,
        'counts': counts.tolist(),
        'phis': phis.tolist(),
        'K': K, 'N': N, 'b': b, 'T': T,
        'zeta': zeta, 'gamma': gamma, 'eta': eta,
        'recording': rec_df,
        'utterances': utter,
        'stats': stats_df,
        'datapoints': datapoints,
        'datascale': datascale,
        'hazard': hazard,
        'record': record,
        'record_utterances': record_utterances
    }

#####################################################################

def save_BNG_simulation(results, base, name):
    """
    Export a simulation run to a directory. It creates a subdirectory
    with the following files:
        - {name}-interactions.csv.gz: Pandas dataframe with all interactions
        - {name}-stats.csv.gz: Pandas dataframe with all statistics
        - {name}-utterances-*.txt.gz: three files describing a CSR sparse matrix
        - {name}-params.json: json file with all parameters of this simulation
    """
    
    # Make a directory with name `name` 
    base = os.path.join(base, name)
    if not os.path.exists(base):
        os.makedirs(base)
    basename = os.path.join(base, name)
    
    if results['record_utterances']:
        save_csr(results['utterances'], basename + '-utterances')
    
    if results['record']:
        int_df = results['recording']
        int_df.to_csv('{}-recording.csv.gz'.format(basename), compression='gzip')
    
    stats_df = results['stats']
    stats_df.to_csv('{}-stats.csv.gz'.format(basename), compression='gzip')
    
    params = {}
    for k, v in results.items():
        if k in ['utterances', 'recording', 'stats']: continue
        
        if type(v) == np.ndarray:
            params[k] = v.tolist()
        else:
            params[k] = v
    json.dump(params, open('{}-params.json'.format(basename), 'w'))

def load_BNG_simulation(directory, params_only=False):
    """Load a simulation
    
    Args:
        directory: path to the directory containing all simulation results

    Returns:
        A dictionary with all simulation data
    """
    name = os.path.basename(directory)
    basename = os.path.join(directory, name)
    
    # Load params 
    results = json.load(open('{}-params.json'.format(basename), 'r'))
    for k, v in results.items():
        if type(v) == list:
            results[k] = np.array(v)

    if params_only == False:
        results['stats'] = pd.read_csv('{}-stats.csv.gz'.format(basename), index_col=0)
        
        if results['record']:
            results['recording'] = pd.read_csv('{}-recording.csv.gz'.format(basename), index_col=0)
        
        if results['record_utterances']:
            results['utterances'] = load_csr(basename+'-utterances', (results['T'], results['K']))

    return results

def sliced_psi(xs, slice_size):
    """Computes the empirical distribution of utterances (psi) uttered in
    different time-slices [0, ..., t), [t, ..., 2t), ..., [T-t, ..., T).
    That is, it computes psi^(0:t), psi^(t:2t), psi^(2t, 3t) and so on.
    If these distributions are all similar, the population most likely
    converges to a stable language. If the divergence is larger, the language
    is most likely unstable
    
    Args:
        xs: a TxK matrix with utterance counts in every round
        binsize: the size of the slices
        
    Returns:
        a (T//slice_size)xK matrix with the empirical distribution psi over
        words in every slice. 
    """
    T, K = xs.shape
    slice_size = min(slice_size, T)
    slices = T // slice_size
    
    # Compute the empirical distribution on every slice
    distributions = np.zeros((slices, K))
    for i, start in enumerate(np.arange(T-slices*slice_size, T, slice_size)):
        end = start + slice_size
        vs = xs[start:end].sum(axis=0)
        distributions[i,:] = normalize(vs)
    
    return JSD(distributions), distributions

def analyze_BNG_simulation_runs(fn, runs, burn=10000, firstrun=1):
    """
    Analyzes simulation runs by collecting various statistics:
        - JSD(\phi_1, ... \phi_N): divergence between agents 
            distributions at the end of the simulation
        - JSD(alpha, \bar\phi): divergence between mean dist
            and prior at the end of the simulation
        - H(\bar\phi): entropy of mean distribution at the end
        - std(JSD(\phi_1^(t), ... \phi_N^(t))): standard deviation of
            div. of agent distributions over time
        - std(JSD(alpha, \bar\phi^(t))): std of mean agent distribution 
            and prior over time
        - pearson_r: Pearson correlation coefficient between log(t) and
            log(JSD(phi_i)). If the distributions converge you expect
            a linear decline.
        - pearson_r_p_value: p-value of the above correlation
        - JSD(\psi, \bar\phi): divergence between time-average (of all
            words uttered) and mean agent distribution at the end
        - H(psi): entropy of time-average at the end
        - JSD(psi, alpha): divergence between time average and prior 
            at the end of the simulation
        - JSD(psi^(0:t), psi(t:2t), ... psi^(T-t:T)): divergence between
            time-averages of slices of the entire simulation.
    
    Besides, we collect divergences averaged over all runs of:
        - JSD(\phi_1, ... \phi_N)
        - JSD(\phi_1, ... \phi_N)
        - H(\bar\phi)
        
    All this is returned in two pandas dataframes; one with the statistics
    listed first, for every run, and one with the divergences over time
    averaged across runs.
    """

    # Load shared params
    params = load_BNG_simulation(fn.format(firstrun), params_only=True)
    alpha_ps = normalize(params['alpha'])
    D = len(params['datapoints'])

    # Collect all stats in a single matrix
    divergences = np.zeros((runs, D, 4))

    # Prepare a array with all statistics
    stats_columns = [
        'jsd_phis',
        'jsd_alpha_mphi',
        'entropy_mphi',
        'std_jsd_phis',
        'std_alpha_mphi',
        'pearson_r',
        'pearson_r_p_value',
        'jsd_mphi_psi',
        'entropy_psi',
        'jsd_psi_alpha',
        'jsd_psi_slices'
    ]
    all_stats = np.zeros((runs, len(stats_columns)))

    # Burn in 
    burn_idx = np.where(params['datapoints'] > burn)[0].min()

    for r, run in enumerate(range(firstrun, firstrun+runs)):
        base = fn.format(run)
        name = os.path.basename(base)
        print('Run {:0>4}/{:0>4} ({})'.format(runs, run, name))

        results = load_BNG_simulation(base)

        # Store all divergences
        divs = results['stats'].as_matrix()
        divergences[r,:,:] = divs

        # Pearson correlation between log(t) and log(jsd(*phi))
        jsd_phis = results['stats']['JSD(*phis)']
        pearson_r, prob_r = pearsonr(np.log10(jsd_phis.index[burn_idx:]), 
                     np.log10(jsd_phis.values[burn_idx:]))

        # Compute JSD of psi^(T) and alpha
        psi_T = normalize(results['utterances'].sum(axis=0))
        jsd_psi_T_alpha = JSD(join(psi_T, alpha_ps, cols=params['K']))

        # JSD between psi across slices
        jsd_psi_slices, _ = sliced_psi(results['utterances'], 2000)

        # JSD between phi^(T) and psi^(T)
        mphi_T = results['phis'].mean(axis=0)
        jsd_psi_T_mphi_T = JSD(join(psi_T, mphi_T, cols=params['K']))

        # Save all those stats
        all_stats[r,:] = (
            results['stats']['JSD(*phis)'].iloc[-1],
            results['stats']['JSD(alpha, mean(*phis))'].iloc[-1],
            results['stats']['H(mean(*phi))'].iloc[-1],
            results['stats']['JSD(alpha, *phis)'].iloc[burn_idx:].std(),
            results['stats']['JSD(alpha, mean(*phis))'].iloc[burn_idx:].std(),
            pearson_r, 
            prob_r, 
            jsd_psi_T_mphi_T,
            entropy(psi_T),
            jsd_psi_T_alpha,
            jsd_psi_slices)

    # Store means and stds in a dataframe
    means = divergences.mean(axis=0)
    std = divergences.std(axis=0)
    _cols = results['stats'].columns
    columns = pd.MultiIndex.from_arrays([
        list(flatten([[c]*2 for c in _cols])),
        ['mean', 'std'] * len(_cols)])
    divergences_df = pd.DataFrame(np.concatenate([means, std], axis=1),
          index=params['datapoints'], columns=columns)

    # Store all statistics in a dataframe
    stats_df = pd.DataFrame(all_stats, columns=stats_columns)
    stats_df.index.name='run'
    
    return divergences_df, stats_df, divergences

def get_pis(K):
    pis = {}
    m = int(K/2)

    pi = np.ones(K)
    pis['flat'] = pi/pi.sum()

    pi = np.arange(1,K+1)
    pis['stair_up'] = pi/pi.sum()

    pi = K - np.arange(K)
    pis['stair_down'] = pi/pi.sum()

    pi = np.ones(K)*0.0001
    pi[m:] = 1
    pis['upper_half'] = pi/pi.sum()

    pi = np.ones(K)*0.0001
    pi[:m] = 1
    pis['lower_half'] = pi/pi.sum()

    pi = np.concatenate((np.arange(1, m+1), (m - np.arange(m))))
    pis['peak'] = pi/pi.sum()

    pi = np.ones(K)
    pi[:int(K/4)] = 10
    pi[-int(K/4):] = 5
    pi[-int(K/4):-int(K/4)+2] = 10 
    pis['gappy'] = pi/pi.sum()

    return pis

#####################################################################
#####################################################################
#####################################################################


if __name__ == '__main__':
    import argparse

    # Define all command line arguments
    parser = argparse.ArgumentParser()

    # Simulation parameters
    parser.add_argument('--runs', type=int, default=1,
        help='Number of runs (defaults to 1)')
    parser.add_argument('--firstrun', type=int, default=1,
        help='Run from which to start (handy if you later add more runs)')
    parser.add_argument('--name', type=str, required=True,
        help='Unique name for this simulation run')
    parser.add_argument('--out', type=str, default='results',
        help='Directory to store results in (Default: results/)')

    # Standard parameters
    parser.add_argument('--T', type=int, required=True,
        help='Number of rounds to play')
    parser.add_argument('--N', type=int, required=True,
        help='Number of agents in the population')
    parser.add_argument('--K',  type=int, required=True,
        help='Number of words (or categories)')
    parser.add_argument('--b', type=int, default=1,
        help='Bottleneck size; number of words uttered in every interaction (Default: 1)')

    # Simulation type parameters
    parser.add_argument('--eta',  required=True,
        help='Strategy parameter for selecting languages during production. \
            interpolates between samplers (eta=1) and MAP (eta="inf")')
    parser.add_argument('--zeta',  required=True,
        help='Strategy parameter for selecting languages during production. \
            interpolates between samplers (zeta=1) and MAP (zeta="inf")')
    parser.add_argument('--gamma',  required=True,
        help='Life expectancy parameter; interpolates between iterated learning (gamma = 1) \
        and a naming game (zeta = "inf")')
    parser.add_argument('--chain', type=int, default=True,
        help='Organise agents in a chain (i.e., random walk through the population)? \
        (Default: True)')
    parser.add_argument('--hazard', type=str, default='weibull',
        help='Type of hazard function: "weibull", "geometric" or "deterministic" (Default: weibull)')

    # Prior
    parser.add_argument('--beta',  type=float, default=1, 
        help='Strength parameter of the hyperprior alpha.')
    parser.add_argument('--pi',  type=str, default='gappy',
        help='Shape of the prior. Possible values are "flat", "stair_up", "stair_down", \
        "upper_half", "lower_half", "peak" and "gappy" (the default)')
    
    # Data collection
    parser.add_argument('--datapoints',  type=int, default=500,
        help='Number of points at which to collect statistics (Default: 500) ')
    parser.add_argument('--datascale', type=str, default='log',
        help='Scale of the statistics. Possible values: "log" and "linear" (Default: log) \
        With a logarithmic datascale, statistics are collected at timesteps that are \
        linearly spaced on a logarithmic Scale.')
    parser.add_argument('--record',  type=int, default=1,
        help='Record the exact history of the game, i.e., the speaker, hearer \
        their ages and so on? (Default: 1 (yes))')
    parser.add_argument('--recordutterances',  type=int, default=1,
        help='Record the words uttered in every round? (Default: 1 (yes))')

    args = parser.parse_args()
    
    if os.path.isdir(args.out) == False:
        raise NotADirectoryError(f'The output directory "{args.out}" could not be found.')
    
    # Hyperparameters alpha
    pis = get_pis(args.K);
    alpha = pis[args.pi] * args.beta

    setup = dict(
        alpha=alpha,
        N=args.N,
        K=args.K,
        T=args.T,
        b=args.b,
        eta=args.eta,
        zeta=args.zeta, 
        gamma=args.gamma, 
        chain=args.chain == 1, 
        hazard=args.hazard,
        num_datapoints=args.datapoints, 
        datascale=args.datascale,
        record=args.record == 1,
        record_utterances=args.recordutterances == 1)

    # Time the runs
    times = np.zeros(args.runs + 1)
    times[0] = time.time();

    # And go!
    for r, run in enumerate(range(args.firstrun, args.firstrun + args.runs)):
        r = r+1

        print('> Starting run {:0>3}/{:0>3} (named {:0>3})...'.format(r, args.runs, run))

        results = BNG_simulation(**setup)
        save_BNG_simulation(results, args.out, '{}-run-{:0>4}'.format(args.name, run))

        times[r] = time.time()
        diffs = times[1:r+1] - times[:r]
        avg = diffs.mean()
        print('  Done in {:.2f}s (around {:.2f}s remaining)'.format(diffs[-1], (args.runs-r)*avg))
