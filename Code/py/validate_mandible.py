"""Leave-one-out validation of the skin -> mandible predictor against each held-out case's GT mandible (trimmed target points).

Methods: oracle (own registration), template (FE mandible carried by the skin-only registration warp = what the pipeline did
before the mandible was modelled), mean shape, multi-atlas, pca-ridge (shape_predict.py). Also the chin-tip error:
distance between the lowest 3% of the predicted mandible nodes and the lowest 3% of the GT mandible points.
"""
import csv, json, sys
import numpy as np, pyvista as pv
from scipy.interpolate import RBFInterpolator
import shape_predict as sp
import hanseg_template_registration as h
from validate_shape_predictors import chamfer_hd95

BASE = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/outputs'
FULL, SKINONLY = f'{BASE}/hanseg_template_reg', f'{BASE}/hanseg_template_reg_skinonly'
DROP = ('case_09', 'case_04')


def tip(p): return p[p[:, 2] <= np.percentile(p[:, 2], 3)].mean(0)


def main(drop=DROP):
    z = np.load(f'{sp.DS}/shape_dataset.npz'); cases = [str(c) for c in z['cases']]
    sr, mr = z['skin_reliable'], z['mandible_reliable']
    skin_al, y_al = z['skin'][:, sr], z['mandible'][:, mr]
    fe, names, pid = h.load_fe(); S, _ = h.fe_structures(fe, names, pid)
    Tm, Ts = np.asarray(S['mandible'].points, float)[mr], np.asarray(S['skin'].points, float)
    to_raw = lambda x, i: (x - z['t'][i]) @ z['R'][i]
    keep = [i for i, c in enumerate(cases) if c not in drop]; res = {m: [] for m in ('oracle', 'template', 'mean shape', 'multi-atlas', 'pca-ridge')}
    for j, i in enumerate(keep):
        tr = [k for k in keep if k != i]; c = cases[i]
        reg = np.asarray(pv.read(f'{SKINONLY}/{c}/{c}_skin_registered.vtp').points)
        idx = np.where(sr)[0]; idx = idx[np.linspace(0, len(idx) - 1, 2500).astype(int)]
        tmpl = RBFInterpolator(Ts[idx], reg[idx], kernel='thin_plate_spline', smoothing=1e-3)(Tm)
        preds = {'oracle': y_al[i], 'template': tmpl @ z['R'][i].T + z['t'][i], 'mean shape': y_al[tr].mean(0),
                 'multi-atlas': sp.multi_atlas(skin_al[tr], y_al[tr], skin_al[i])[0],
                 'pca-ridge': sp.pca_ridge_predict(sp.pca_ridge_fit(skin_al[tr], y_al[tr]), skin_al[i])[0]}
        tg = np.load(f'{FULL}/{c}/{c}_mandible_target_trimmed.npy'); gtip = tip(tg)
        for m, p in preds.items():
            p = to_raw(p, i); ch, hd = chamfer_hd95(p, tg)
            res[m].append(dict(case=c, chamfer=ch, hd95=hd, tip_err=float(np.linalg.norm(tip(p) - gtip)), tip_dz=float(tip(p)[2] - gtip[2])))
        print(f'  {j + 1}/{len(keep)} {c}', file=sys.stderr, flush=True)
    print(f'== mandible, leave-one-out, {len(keep)} held-out cases (median; worst) ==')
    print(f'{"method":12s} | {"chamfer mm":>16s} {"hd95 mm":>9s} | {"chin-tip error mm":>20s} | tip dz (signed, median)')
    for m, rows in res.items():
        f = lambda k: f'{np.median([r[k] for r in rows]):.2f} ({max(r[k] for r in rows):.1f})'
        print(f'{m:12s} | {f("chamfer"):>16s} {np.median([r["hd95"] for r in rows]):9.1f} | {f("tip_err"):>20s} | {np.median([r["tip_dz"] for r in rows]):+.1f}')
    d = np.array([a['chamfer'] for a in res['pca-ridge']]); mn = np.array([a['chamfer'] for a in res['mean shape']]); tm = np.array([a['chamfer'] for a in res['template']])
    print(f'pca-ridge better than mean shape in {(d < mn).sum()}/{len(d)}, than template in {(d < tm).sum()}/{len(d)}')
    json.dump(res, open(f'{sp.DS}/validation_mandible.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
