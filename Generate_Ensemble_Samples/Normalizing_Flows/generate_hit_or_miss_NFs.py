import os
import json
import torch
import numpy as np
import argparse
import sys
import gc

NF_UTILS = os.path.join(os.path.dirname(__file__), '..', '..', 'Train_Ensembles', 'Train_Models')
sys.path.insert(0, os.path.abspath(NF_UTILS))
from utils_flows import make_flow

parser = argparse.ArgumentParser()
parser.add_argument('--trial_dir', type=str, required=True,
                    help='Dir with f_i.pth and architecture_config.json (Train_Ensembles output).')
parser.add_argument('--w_path', type=str, required=True,
                    help='Path to fitted WiFi weights w_i_fitted.npy.')
parser.add_argument('--out_dir', type=str, required=True,
                    help='Base output directory.')
parser.add_argument('-n', '--ngenerate', type=int, required=True,
                    help='Number of events to generate per seed.')
parser.add_argument('-s', '--seed', type=int, required=True,
                    help='Random seed (also used as output filename index).')
parser.add_argument('--tail_bound', type=float, default=3.0,
                    help='Sampling bounds: [-tail_bound, tail_bound] per dimension.')
args = parser.parse_args()

# Auto-name output subdir from wifi weights dir (consistent with kernel convention)
wifi_name = os.path.basename(os.path.dirname(os.path.abspath(args.w_path)))
effective_out_dir = os.path.join(args.out_dir, wifi_name)
os.makedirs(effective_out_dir, exist_ok=True)

np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load architecture config and NF models
with open(os.path.join(args.trial_dir, 'architecture_config.json')) as f:
    config = json.load(f)

f_i_statedicts = torch.load(os.path.join(args.trial_dir, 'f_i.pth'), map_location=device)
print('Loaded', len(f_i_statedicts), 'models')

f_i_models = []
for state_dict in f_i_statedicts:
    flow = make_flow(
        num_layers=config['num_layers'],
        hidden_features=config['hidden_features'],
        num_bins=config['num_bins'],
        num_blocks=config['num_blocks'],
        num_features=config['num_features'],
    )
    flow.load_state_dict(state_dict)
    flow.eval()
    f_i_models.append(flow)
    del flow
    gc.collect()

# Load WiFi weights from file
w_i_fitted = torch.tensor(np.load(args.w_path), dtype=torch.float64)
print('WiFi weights shape:', w_i_fitted.shape, 'sum:', round(w_i_fitted.sum().item(), 6))


def _eval_ensemble(x1, x2):
    x = torch.stack([x1, x2], dim=1)
    probs_list = []
    with torch.no_grad():
        for model in f_i_models:
            model = model.to(x.device)
            model_probs = []
            for i in range(0, len(x), 5000):
                logp = model.log_prob(x[i:i+5000].to(x.device))
                model_probs.append(torch.exp(logp))
            probs_list.append(torch.cat(model_probs).detach())
            model.to('cpu')
            torch.cuda.empty_cache()
    probs = torch.stack(probs_list, dim=1)
    w = w_i_fitted.to(probs.device)
    return (probs * w).sum(dim=1)


def hit_or_miss_2d(tb, N_events, max_attempts=1_000_000_000):
    f_max = 2.0
    print('f_max:', f_max, 'bounds: [%.1f, %.1f]' % (-tb, tb))
    accepted = []
    total_hits = 0
    total_attempts = 0
    batch_size = max(N_events // 2, 10000)
    while total_hits < N_events and total_attempts < max_attempts:
        x1 = torch.empty(batch_size, device=device).uniform_(-tb, tb)
        x2 = torch.empty(batch_size, device=device).uniform_(-tb, tb)
        y  = torch.empty(batch_size, device=device).uniform_(0, f_max)
        f_vals = _eval_ensemble(x1, x2)
        f_vals = torch.where(f_vals > 0, f_vals, torch.zeros_like(f_vals))
        hits = torch.stack([x1[y < f_vals], x2[y < f_vals]], dim=1)
        accepted.append(hits)
        total_hits += hits.shape[0]
        total_attempts += batch_size
        print(f'total hits: {total_hits}')
    if not accepted:
        raise RuntimeError('No events accepted — check density function.')
    return torch.cat(accepted)[:N_events]


samples = hit_or_miss_2d(args.tail_bound, N_events=args.ngenerate)
out_path = os.path.join(effective_out_dir, 'seed%i.npy' % args.seed)
np.save(out_path, samples.cpu().numpy())
print('Saved %i events to %s' % (len(samples), out_path))
