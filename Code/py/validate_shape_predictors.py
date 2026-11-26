"""Leave-one-out validation of the skin -> skull/brain predictors against each held-out case's ORIGINAL GT surfaces.

Predictions are mapped back into the held-out case's GT frame and scored against the trimmed GT target points saved
by the registration run (same surfaces the registration was scored on). Methods:
  oracle        the case's own registration (saw its GT; the floor any predictor can reach)
  template      FE template carried by the skin-only registration warp (what the pipeline does without a predictor)
  mean shape    training-fold mean
  multi-atlas, pca-ridge  (shape_predict.py)
"""
import csv, json, os, sys
import numpy as np, pyvista as pv
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import shape_predict as sp
import hanseg_template_registration as h

BASE = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/outputs'
FULL, SKINONLY = f'{BASE}/hanseg_template_reg', f'{BASE}/hanseg_template_reg_skinonly'
SETS = {'38 cases (excl. 09, 04)': ('case_09', 'case_04'), '36 cases (also excl. 23, 32: gate fail)': ('case_09', 'case_04', 'case_23', 'case_32')}


def chamfer_hd95(pred, tgt):
    a, b = cKDTree(tgt).query(pred)[0], cKDTree(pred).query(tgt)[0]
    return a.mean() + b.mean(), max(np.percentile(a, 95), np.percentile(b, 95))


def depth(pts, skin):                       # distance from each point to the (input) skin surface
    return cKDTree(skin).query(pts)[0]


_T = None
def _template():
    global _T
    if _T is None:
        fe, names, pid = h.load_fe(); s, _ = h.fe_structures(fe, names, pid)
        _T = {k: np.asarray(v.points, float) for k, v in s.items()}
    return _T


def template_baseline(z, i, skin_rel, skull_rel, brain_rel):
    """TPS from template skin nodes -> this case's registered skin nodes, applied to template skull/brain nodes."""
    T = _template()
    case = str(z['cases'][i])
    reg = np.asarray(pv.read(f'{SKINONLY}/{case}/{case}_skin_registered.vtp').points)
    idx = np.where(skin_rel)[0]; idx = idx[np.linspace(0, len(idx) - 1, 2500).astype(int)]
    tps = RBFInterpolator(T['skin'][idx], reg[idx], kernel='thin_plate_spline', smoothing=1e-3)
    return np.concatenate([tps(T['skull'][skull_rel]), tps(T['brain'][brain_rel])])


def main():
    z = np.load(f'{sp.DS}/shape_dataset.npz'); cases_all = [str(c) for c in z['cases']]
    sr, kr, br = z['skin_reliable'], z['skull_reliable'], z['brain_reliable']
    skin_al = z['skin'][:, sr]; y_al = np.concatenate([z['skull'][:, kr], z['brain'][:, br]], axis=1)
    ns = int(kr.sum()); to_raw = lambda x, i: (x - z['t'][i]) @ z['R'][i]
    tgt = lambda c, k, d=FULL: np.load(f'{d}/{c}/{c}_{k}_target_trimmed.npy')
    results = {}
    for set_name, drop in SETS.items():
        keep = [i for i, c in enumerate(cases_all) if c not in drop]; res = {m: [] for m in ('oracle', 'template', 'mean shape', 'multi-atlas', 'pca-ridge')}
        extra = []
        for j, i in enumerate(keep):
            tr = [k for k in keep if k != i]; c = cases_all[i]
            preds = {'oracle': (y_al[i], None), 'template': (template_baseline(z, i, sr, kr, br) @ z['R'][i].T + z['t'][i], None),
                     'mean shape': (y_al[tr].mean(0), None), 'multi-atlas': sp.multi_atlas(skin_al[tr], y_al[tr], skin_al[i]),
                     'pca-ridge': sp.pca_ridge_predict(sp.pca_ridge_fit(skin_al[tr], y_al[tr]), skin_al[i])}
            tg = {'skull': tgt(c, 'skull'), 'brain': tgt(c, 'brain')}; skin_in = to_raw(skin_al[i], i)
            for m, (p, sig) in preds.items():
                p = to_raw(p, i); row = {'case': c}
                for part, sl in (('skull', slice(0, ns)), ('brain', slice(ns, None))):
                    row[f'{part}_chamfer'], row[f'{part}_hd95'] = chamfer_hd95(p[sl], tg[part])
                row['skull_depth'] = float(np.median(depth(p[:ns], skin_in)))
                res[m].append(row)
            gt_depth = float(np.median(depth(tg['skull'], tgt(c, 'skin', SKINONLY))))
            for r in (res[m][-1] for m in res): r['skull_depth_err'] = abs(r['skull_depth'] - gt_depth)
            sig = preds['pca-ridge'][1]
            e = np.linalg.norm(preds['pca-ridge'][0] - y_al[i], axis=1)
            extra.append((float(sig.mean()), float(e.mean())))
            print(f'  [{set_name[:9]}] {j + 1}/{len(keep)} {c}', file=sys.stderr, flush=True)
        results[set_name] = res
        sg = np.array(extra); rho = float(np.corrcoef(np.argsort(np.argsort(sg[:, 0])), np.argsort(np.argsort(sg[:, 1])))[0, 1])
        print(f'\n== training set: {set_name}  ({len(keep)} held-out cases) ==')
        print(f'{"method":12s} | {"skull chamfer":>14s} {"skull hd95":>11s} | {"brain chamfer":>14s} {"brain hd95":>11s} | skin->skull depth err')
        for m, rows in res.items():
            f = lambda k: f'{np.median([r[k] for r in rows]):.2f} (worst {max(r[k] for r in rows):.1f})'
            print(f'{m:12s} | {f("skull_chamfer"):>14s} {np.median([r["skull_hd95"] for r in rows]):11.1f} | {f("brain_chamfer"):>14s} {np.median([r["brain_hd95"] for r in rows]):11.1f} | {np.median([r["skull_depth_err"] for r in rows]):.2f} mm')
        for part in ('skull', 'brain'):
            d = np.array([a[f'{part}_chamfer'] for a in res['pca-ridge']]) - np.array([a[f'{part}_chamfer'] for a in res['mean shape']])
            t = np.array([a[f'{part}_chamfer'] for a in res['pca-ridge']]) - np.array([a[f'{part}_chamfer'] for a in res['template']])
            print(f'  {part}: pca-ridge better than mean shape in {(d < 0).sum()}/{len(d)} cases, than template in {(t < 0).sum()}/{len(t)}')
        print(f'  uncertainty: rank correlation between predicted sigma and actual error across cases = {rho:.2f}')
    json.dump(results, open(f'{sp.DS}/validation_results.json', 'w'), indent=1, default=float)
    first = next(iter(results.values()))
    with open(f'{sp.DS}/validation_38cases.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['method', *first['oracle'][0].keys()])
        for m, rows in first.items():
            for r in rows: w.writerow([m, *r.values()])


if __name__ == '__main__':
    main()
