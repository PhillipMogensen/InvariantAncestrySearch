import numpy as np
from scipy.stats import ttest_ind, bartlett
from sklearn.linear_model import LinearRegression
from multiprocess import Pool, cpu_count
# from torch import outer
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from argparse import ArgumentParser
from itertools import chain

#### Loading of data ####
observation, intervention, intervention_pos = (np.genfromtxt(
    f'real_world_analysis/data/Kemmeren{spec}.csv', delimiter='\t', skip_header=1
  ) for spec in ['Obs', 'Int', 'IntPos']
)
# Fix indentation starting at 1
intervention_pos = (intervention_pos - 1).astype(int)

n_obs = observation.shape[0]
n_int = intervention.shape[0]
n_gene = intervention.shape[1]

#### Global defitions ####
X = np.concatenate([observation, intervention])
E = np.repeat([0, 1], [n_obs, n_int])


def get_idx(gene_x):
  """
  Return the experiment id of the intervention on gene_x. E.g. if 3rd intervention is on gene number 3687, get_idx(3687) -> 3
  """
  if gene_x is None: return None
  else: 
    positions = np.where(gene_x == intervention_pos)[0]
    return positions[0] if len(positions) > 0 else None

def get_2d(X): 
  return X if len(X.shape) > 1 else X.reshape(-1, 1)


#### Function defitions ####
def is_true_positive(gene_x, gene_y, threshold=0.01):
  """
  Function to check whether gene_x is a true cause of gene_y, in the sense that
  when intervening on gene_x, gene_y lies in the upper or lower 1% tail of its
  observational distribution. 
  """
  quantiles_y = np.quantile(observation[:, gene_y], q = [threshold, 1-threshold])
  intervention_effect = intervention[get_idx(gene_x), gene_y]
  return (intervention_effect <= quantiles_y[0]) | (intervention_effect >= quantiles_y[1])

def icp_test(e, x, y, covar):
  """
  Tests the specified set of covariates for invariance, as described in the
  original ICP paper (method II)
  """
  if len([x for x in covar if x is not None]) == 0 or (covar is None):
    resid = y
  else:
    # Perform linear regression and get residuals
    x2 = get_2d(x[:, covar])
    model = LinearRegression(fit_intercept = False).fit(X=x2, y=y)
    resid = y - model.predict(x2)

  # Split residuals by environments
  envs = tuple(resid[e == e_] for e_ in np.unique(e))
  
  # Conduct p-vals from t- and bartlett-tests and Bonferroni correct
  return 2*np.min([ttest_ind(*envs)[1], bartlett(*envs)[1]])

def test_gene(gene_x, gene_y, test_empty=False):
  """
  Tests whether gene_x is an invariant predictor of gene_y
  """
  self_idx = [(n_obs + idx) for idx in [get_idx(gene_x), get_idx(gene_y)] if idx is not None]
  out = icp_test(
    np.delete(E, self_idx, axis=0),
    np.delete(X, self_idx, axis=0),
    np.delete(X, self_idx, axis = 0)[:, gene_y],
    [int(gene_x) if (gene_x is not None and not test_empty) else None]
  )
  return out

def get_ancestors(gene_y, alpha_empty=1e-12, alpha_gene=0.1):
    # Test if empty set invariant
    results = []
    
    empty_p_val = test_gene(None, gene_y)

    if empty_p_val < alpha_empty: # Empty set non-invariant, loop over all predictors
        for gene_x in intervention_pos:
            if gene_x != gene_y:
                gene_x_p_val = test_gene(gene_x, gene_y)
                if gene_x_p_val > alpha_gene:
                    results.append({'gene_y': gene_y, 'gene_x': gene_x, 'p_val': gene_x_p_val, 'TP': is_true_positive(gene_x, gene_y)})
    else: # Empty set is invariant
        results.append({'gene_y': gene_y, 'gene_x': -1, 'p_val': empty_p_val})

    return results

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('start', type=int, help='starting gene for loop over target genes')
  parser.add_argument('end', type=int, help='ending gene for loop over target genes')
  parser.add_argument('cpus', type=int, default=None, help='number of cpus to use for processing')

  args = parser.parse_args()

  n_cpus = min(args.cpus, cpu_count()-1) if args.cpus is not None else cpu_count()-1

  results = pd.DataFrame(list(chain(*tqdm(Pool(n_cpus-1).imap_unordered(get_ancestors, range(args.start, args.end)), total=args.end-args.start))))
  results.to_pickle(f"output/study4_gene_expression/sub_results/saved_{args.start:04}-{args.end:04}.pkl")



