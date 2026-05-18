"""
Wald chi-square sanity test for the wifi pipeline.

Tests "is q sufficient?" using the wifi-fitted weights w_hat against the
no-reweighting point w = 0 (which makes r(x;w) = 1 and so r*q reduces to q):

    T = w_hat^T Sigma_w^{-1} w_hat   ~_{H_0}   chi^2_{K+1}

Under H_0 (data ~ q in span{f_i}), w* = 0 and T is chi-square distributed
with K+1 degrees of freedom (counting the bias and the K basis logits).

This is NOT the goodness-of-fit test for r_hat * q. It only tests whether
the data is consistent with q itself, in the directions the wifi basis can
see. We use it as a sanity check that the pipeline produces something
non-trivial when q is clearly wrong (e.g., 2d_gmm_skew vs Gaussian).
"""

import torch
from scipy.stats import chi2


def wald_chi2(w_hat, Sigma_w, ridge_rel=1e-8):
    """
    Compute the Wald statistic T = w_hat^T Sigma_w^{-1} w_hat and its p-value
    under chi^2_{dim(w_hat)}.

    Parameters
    ----------
    w_hat   : (K+1,) tensor
    Sigma_w : (K+1, K+1) tensor — should be a covariance matrix of w_hat.
    ridge_rel : relative Tikhonov ridge before inversion.

    Returns
    -------
    dict with keys: T (float), dof (int), p_value (float),
                    Sigma_inv (tensor)
    """
    w = w_hat.double()
    S = Sigma_w.double()
    K1 = w.shape[0]
    eps = ridge_rel * (torch.trace(S) / K1)
    S_reg = S + eps * torch.eye(K1, dtype=S.dtype)
    S_inv = torch.linalg.inv(S_reg)
    T = float(w @ S_inv @ w)
    p = float(chi2.sf(T, df=K1))
    return {"T": T, "dof": K1, "p_value": p, "Sigma_inv": S_inv}
