"""Predict skull + brain (FE node order) from the skin alone, with a per-node uncertainty.

Two predictors, both trained on shape_dataset.npz (built by build_shape_dataset.py):
  multi_atlas : kernel-weighted mean of the nearest training cases (skin distance); spread across atlases = sigma
  pca_ridge   : skin PCA coefficients -> ridge -> skull/brain; sigma from leave-one-out residuals of the training fold
Run as a script for leave-one-out evaluation against the mean-shape baseline.
"""
import json, sys
import numpy as np
from scipy.stats import chi2

DS = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/outputs/shape_dataset'
K_SKIN, LAM, K_ATLAS = 8, 0.1, 10      # fixed a priori; the grid below only shows sensitivity
CHI95 = chi2.ppf(0.95, 3)


def load(drop=()):
    z = np.load(f'{DS}/shape_dataset.npz'); keep = np.array([c not in drop for c in z['cases']])
    sk, sl, br = z['skin_reliable'], z['skull_reliable'], z['brain_reliable']
    skin = z['skin'][keep][:, sk]; n = keep.sum()
    y = np.concatenate([z['skull'][keep][:, sl], z['brain'][keep][:, br]], axis=1)
    return z['cases'][keep], skin, y, int(sl.sum())          # y: (n, Nskull+Nbrain, 3)


def load_struct(name, drop=()):
    """Same cases/skin as load(), but one named structure ('mandible', ...) as the target: (cases, skin, y, n_nodes)."""
    z = np.load(f'{DS}/shape_dataset.npz'); keep = np.array([c not in drop for c in z['cases']])
    m = z[f'{name}_reliable']
    return z['cases'][keep], z['skin'][keep][:, z['skin_reliable']], z[name][keep][:, m], int(m.sum())


def multi_atlas(skin_tr, y_tr, skin_q, k=K_ATLAS):
    d = np.sqrt(((skin_tr - skin_q) ** 2).sum(2).mean(1))       # RMS node distance to each atlas
    idx = np.argsort(d)[:k]; w = 1 / d[idx] ** 2; w /= w.sum()
    mu = (w[:, None, None] * y_tr[idx]).sum(0)
    var = (w[:, None] * ((y_tr[idx] - mu) ** 2).sum(2)).sum(0) / 3
    return mu, np.sqrt(var)


def pca_ridge_fit(skin_tr, y_tr, k_skin=K_SKIN, lam=LAM):
    n = len(skin_tr); X = skin_tr.reshape(n, -1); Y = y_tr.reshape(n, -1)
    mx, my = X.mean(0), Y.mean(0)
    _, _, Vx = np.linalg.svd(X - mx, full_matrices=False); Vx = Vx[:k_skin]
    C = (X - mx) @ Vx.T
    A = C.T @ C + lam * np.trace(C.T @ C) / k_skin * np.eye(k_skin)
    B = np.linalg.solve(A, C.T @ (Y - my))                        # (k, F)
    H = C @ np.linalg.solve(A, C.T)                               # hat matrix: LOO residual = r / (1 - h_ii)
    loo = (Y - my - C @ B) / (1 - np.diag(H))[:, None]
    sigma = np.sqrt((loo ** 2).reshape(n, -1, 3).sum(2).mean(0) / 3)
    off = (X - mx) - C @ Vx                                        # skin part the k modes cannot explain
    r = np.sqrt((off ** 2).mean(1))
    return dict(mx=mx, my=my, Vx=Vx, B=B, sigma=sigma, shape=y_tr.shape[1:], Ainv=np.linalg.inv(A),
                res_mu=r.mean(), res_sd=r.std() + 1e-9)


def query_stats(m, skin_q):
    """Case-specific quantities of a query skin: leverage in the fitted skin space and how far its
    unexplained skin residual sits above the training residuals."""
    x = skin_q.reshape(1, -1) - m['mx']; c = x @ m['Vx'].T
    h = float(c @ m['Ainv'] @ c.T)
    r = float(np.sqrt(((x - c @ m['Vx']) ** 2).mean()))
    return c, h, (r - m['res_mu']) / m['res_sd']


def pca_ridge_predict_old(m, skin_q, case_specific=False):
    c, h, z = query_stats(m, skin_q)
    sig = m['sigma'] * (np.sqrt(1 + h) if case_specific else 1.0)
    return (m['my'] + c @ m['B']).reshape(m['shape']), sig


def pca_ridge_predict(m, skin_q, calib=None):
    """calib=None: population sigma. calib=(a, b): sigma * b * exp(a * max(z, 0)), z = how far the query skin's
    unexplained shape sits above the training skins' (case-specific)."""
    c, h, z = query_stats(m, skin_q)
    s = 1.0 if calib is None else calib[1] * np.exp(calib[0] * max(z, 0.0))
    return (m['my'] + c @ m['B']).reshape(m['shape']), m['sigma'] * s


def fit_calibration(skin, y, **kw):
    """Fit (a, b) by maximum likelihood on the training fold's own leave-one-out predictions."""
    from scipy.optimize import minimize
    n = len(skin); zs, qs = np.zeros(n), np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i; m = pca_ridge_fit(skin[tr], y[tr], **kw)
        c, _, zs[i] = query_stats(m, skin[i])
        e = np.linalg.norm((m['my'] + c @ m['B']).reshape(m['shape']) - y[i], axis=1)
        qs[i] = np.mean(e ** 2 / np.maximum(m['sigma'], 1e-6) ** 2)
    zp = np.maximum(zs, 0)
    nll = lambda p: np.sum(3 * (np.log(p[1]) + p[0] * zp) + 0.5 * qs / (p[1] * np.exp(p[0] * zp)) ** 2)
    r = minimize(lambda p: nll([p[0], np.exp(p[1])]), [0.2, 0.0], bounds=[(0, 2), (-2, 2)])
    return float(r.x[0]), float(np.exp(r.x[1]))


def errors(pred, truth, sigma, n_skull):
    e = np.linalg.norm(pred - truth, axis=1); out = {}
    for name, sl in (('skull', slice(0, n_skull)), ('brain', slice(n_skull, None))):
        d = e[sl]; out[name] = dict(mean=d.mean(), p95=np.percentile(d, 95), max=d.max())
        if sigma is not None: out[name]['cover95'] = float((d ** 2 / np.maximum(sigma[sl], 1e-6) ** 2 < CHI95).mean())
    return out


def loo(cases, skin, y, n_skull, method, **kw):
    rows = []
    for i in range(len(cases)):
        tr = np.arange(len(cases)) != i
        if method == 'mean': pred, sig = y[tr].mean(0), None
        elif method == 'atlas': pred, sig = multi_atlas(skin[tr], y[tr], skin[i])
        else: pred, sig = pca_ridge_predict(pca_ridge_fit(skin[tr], y[tr], **kw), skin[i])
        rows.append(errors(pred, y[i], sig, n_skull))
    return rows


def summ(rows, part, key):
    v = np.array([r[part][key] for r in rows]); return float(np.median(v)), float(v.max())


if __name__ == '__main__':
    cases, skin, y, ns = load(drop=('case_09',))
    print(f'{len(cases)} cases; leave-one-out; error = distance to the held-out case\'s registered node (mm)')
    res = {}
    for label, method, kw in (('mean shape (baseline)', 'mean', {}), ('multi-atlas', 'atlas', {}), ('pca-ridge', 'ridge', {})):
        rows = loo(cases, skin, y, ns, method, **kw); res[label] = rows
        line = f'{label:22s}'
        for part in ('skull', 'brain'):
            m, mx = summ(rows, part, 'mean'); p, _ = summ(rows, part, 'p95'); line += f' | {part}: mean {m:.2f} (worst case {mx:.2f}) p95 {p:.2f}'
            if 'cover95' in rows[0][part]: line += f' cover95 {np.median([r[part]["cover95"] for r in rows]):.2f}'
        print(line)
    print('\nsensitivity of pca-ridge (median over cases of mean node error, skull / brain):')
    for k in (3, 5, 8, 11):
        print(f'  k_skin={k:2d}', *[f'lam={lam}: {summ(r, "skull", "mean")[0]:.2f}/{summ(r, "brain", "mean")[0]:.2f}'
                                    for lam in (0.01, 0.1, 1.0) for r in [loo(cases, skin, y, ns, "ridge", k_skin=k, lam=lam)]])
    json.dump({k: v for k, v in res.items()}, open(f'{DS}/loo_results.json', 'w'), indent=1, default=float)
