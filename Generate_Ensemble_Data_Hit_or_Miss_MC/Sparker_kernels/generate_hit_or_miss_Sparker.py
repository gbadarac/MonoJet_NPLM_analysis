import numpy as np
import os, argparse, sys

SPARKER_UTILS = os.path.join(
    os.path.dirname(__file__),                # .../Generate_Ensemble_Data_Hit_or_Miss_MC/Sparker_kernels
    '..', '..', 'Train_Ensembles', 'Train_Models', 'Sparker_utils'
)
sys.path.insert(0, os.path.abspath(SPARKER_UTILS))
import ENSEMBLEutils as ens

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_dir', type=str, required=True,
                    help="Dir with config.json and seed*/ histories (Train_Ensembles output).")
parser.add_argument('--w_path', type=str, required=True,
                    help="Path to fitted WiFi weights .npy (e.g. final_weights.npy).")
parser.add_argument('--out_dir', type=str, required=True,
                    help="Output directory for generated samples.")
parser.add_argument('-n', '--ngenerate', type=int, required=True,
                    help="Number of events to generate.")
parser.add_argument('-e', '--nensemble', type=int, required=True,
                    help="Number of ensemble members.")
parser.add_argument('-s', '--seed', type=int, required=True,
                    help="Random seed (also used as output filename index).")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Load ensemble members (same approach as supervisor's script:
# include all kernels including the normalisation component)
centroids_init, coefficients_init, widths_init = [], [], []
for i in range(args.nensemble):
    seed_dir = os.path.join(args.ensemble_dir, 'seed%03d' % i)
    tmp = np.load(os.path.join(seed_dir, 'widths_history.npy'))
    count = -1
    for j in range(tmp.shape[0]):
        if tmp[j][0].sum():
            count += 1
        else:
            print(count)
            break
    centroids_init.append(np.load(os.path.join(seed_dir, 'centroids_history.npy'))[count])
    coefficients_init.append(np.load(os.path.join(seed_dir, 'coeffs_history.npy'))[count])
    widths_init.append(np.load(os.path.join(seed_dir, 'widths_history.npy'))[count])

centroids_init    = np.stack(centroids_init, axis=0)
coefficients_init = np.stack(coefficients_init, axis=0)
coefficients_init = coefficients_init / np.sum(coefficients_init, axis=1, keepdims=True)
widths_init       = np.stack(widths_init, axis=0)

# Load WiFi weights (separate from ensemble_dir, unlike supervisor's setup)
weights_centralv = np.load(args.w_path)

# Sampling bounds matching the target distribution:
#   x1: bimodal Gaussian (modes at -0.70 and -0.30, sigma=0.12)
#   x2: skew-normal (loc=1.0, scale=0.75, alpha=8.0, strong right skew)
# Same bounds as supervisor's 2GMMskew script (identical dataset).
bounds = [(-1.5, 0.5), (0.5, 4.5)]

samples = ens.hit_or_miss_sample_batch(
    seed=args.seed,
    N=args.ngenerate,
    centroids=centroids_init,
    coefficients=coefficients_init,
    widths=widths_init,
    weights=weights_centralv,
    bounds=bounds,
)

out_path = os.path.join(args.out_dir, 'seed%i.npy' % args.seed)
np.save(out_path, samples)
print("Saved %i events to %s" % (len(samples), out_path))
