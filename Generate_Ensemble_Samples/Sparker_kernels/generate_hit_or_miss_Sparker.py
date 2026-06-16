import numpy as np
import os, argparse, sys

SPARKER_UTILS = os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'shared', 'Sparker_utils'
)
sys.path.insert(0, os.path.abspath(SPARKER_UTILS))
import ENSEMBLEutils as ens

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_dir', type=str, required=True,
                    help="Dir with config.json and seed*/ histories (Train_Ensembles output).")
parser.add_argument('--w_path', type=str, required=True,
                    help="Path to fitted WiFi weights w_i_fitted.npy.")
parser.add_argument('--out_dir', type=str, required=True,
                    help="Base output directory for generated samples.")
parser.add_argument('-n', '--ngenerate', type=int, required=True,
                    help="Number of events to generate per seed.")
parser.add_argument('-e', '--nensemble', type=int, required=True,
                    help="Number of ensemble members.")
parser.add_argument('-s', '--seed', type=int, required=True,
                    help="Random seed (also used as output filename index).")
parser.add_argument('--bounds', type=float, nargs=4, required=True,
                    metavar=('X1_MIN', 'X1_MAX', 'X2_MIN', 'X2_MAX'),
                    help="Hit-or-miss sampling bounds per dimension.")
args = parser.parse_args()

# Auto-create a subfolder named after the WiFi weights run (parent dir of w_path),
# so each wifi ensemble gets its own directory under out_dir automatically.
wifi_name = os.path.basename(os.path.dirname(os.path.abspath(args.w_path)))
effective_out_dir = os.path.join(args.out_dir, wifi_name)
os.makedirs(effective_out_dir, exist_ok=True)

# Load ensemble members
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

# Load WiFi weights
weights_centralv = np.load(args.w_path)

x1_min, x1_max, x2_min, x2_max = args.bounds
bounds = [(x1_min, x1_max), (x2_min, x2_max)]

samples = ens.hit_or_miss_sample_batch(
    seed=args.seed,
    N=args.ngenerate,
    centroids=centroids_init,
    coefficients=coefficients_init,
    widths=widths_init,
    weights=weights_centralv,
    bounds=bounds,
)

out_path = os.path.join(effective_out_dir, 'seed%i.npy' % args.seed)
np.save(out_path, samples)
print("Saved %i events to %s" % (len(samples), out_path))
