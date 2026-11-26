"""Repair the genuinely defective hexes of Head_V6.vtu (found with mesh_qa_jacobian), writing Head_V6_repaired.vtu.

Defects: two full hexes with one slightly negative corner (CSF_Cerebrum_R/L, scaled Jacobian -0.045) and four collapsed
wedges with ~zero volume (CSF_Stem_R/L). Only node coordinates move (topology untouched), each node by at most 0.3x the
shortest edge of any element it belongs to, and no neighbouring element may end up worse than min(its old quality, 0.05).
"""
import json, sys
import numpy as np, pyvista as pv
from scipy.optimize import minimize
import mesh_qa_jacobian as Q

SRC = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/mesh_model/Head_V6.vtu'
DST = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/mesh_model/Head_V6_repaired.vtu'
TARGET = 0.05


def main():
    fe = pv.read(SRC); P = np.asarray(fe.points, float).copy(); P0 = P.copy()
    hx, C, coll = Q.hex_connectivity(fe); ep = np.asarray(fe.cell_data['part_id'])[hx]
    eid = np.asarray(fe.cell_data['element_id'])[hx]
    pat = np.array([tuple(np.unique(c, return_index=True)[1]) if k else () for c, k in zip(C, coll)], dtype=object)
    r = Q.assess(fe, P); bad = np.where(r['bad'])[0]

    # reference volume and sign for collapsed elements: siblings = same part, collapsed, not themselves bad
    vol = Q.proxy_volume(P[C]); ref = {}
    for p in np.unique(ep[bad]):
        sib = coll & (ep == p) & ~r['bad'] & (np.abs(vol) > 1e-9)
        ref[p] = (np.sign(np.median(vol[sib])), np.median(np.abs(vol[sib])))
    inc = {}
    for i, row in enumerate(C):
        for n in set(row.tolist()): inc.setdefault(n, []).append(i)

    def q(i, X):                                              # unit-free quality, target >= TARGET
        if coll[i]: s, v = ref[ep[i]]; return s * Q.proxy_volume(X[None])[0] / v
        return Q.min_scaled_jacobian(X[None])[0]

    def quality_all(P_, els): return np.array([q(i, P_[C[i]]) for i in els])

    log = []
    for b in bad:
        nodes = sorted(set(C[b].tolist())); els = sorted({e for n in nodes for e in inc[n]})
        old = quality_all(P, els); floor = np.minimum(old, TARGET)
        L = min(min(d for d in [np.linalg.norm(P[x] - P[y]) for x in set(C[e]) for y in set(C[e]) if x < y] if d > 1e-9) for e in els)
        cap = 0.3 * L

        def obj(d):
            Pn = P.copy(); D = d.reshape(-1, 3)
            for k, n in enumerate(nodes):
                v = D[k]; m = np.linalg.norm(v); Pn[n] = P[n] + (v if m <= cap else v * cap / m)
            qn = quality_all(Pn, els)
            return -np.min(np.minimum(qn - floor + TARGET, 2 * TARGET)) + 1e-3 * np.sum(D ** 2) / cap ** 2

        best = None
        for seed in range(6):
            x0 = np.random.default_rng(seed).normal(scale=0.05 * cap, size=3 * len(nodes))
            res = minimize(obj, x0, method='Powell', options=dict(xtol=1e-3, ftol=1e-6, maxiter=4000))
            if best is None or res.fun < best.fun: best = res
        D = best.x.reshape(-1, 3)
        for k, n in enumerate(nodes):
            v = D[k]; m = np.linalg.norm(v); P[n] = P[n] + (v if m <= cap else v * cap / m)
        new = quality_all(P, els); worst_nb = np.min(new - floor)
        log.append(f'elem {eid[b]} (part {int(ep[b])}): quality {old[els.index(b)]:.3f} -> {new[els.index(b)]:.3f}; '
                   f'max node move {np.linalg.norm(P[nodes] - P0[nodes], axis=1).max():.3f} mm (cap {cap:.2f}); '
                   f'{len(els)} elements touched, worst change vs floor {worst_nb:+.3f}')
    out = fe.copy(); out.points = P; out.save(DST)

    a = Q.assess(out, P, template_pts=P0)
    # collapsed elements are judged by sibling sign here (their own template sign is ~0 for the four repaired ones)
    vol1 = Q.proxy_volume(P[C]); still = [int(eid[b]) for b in bad if (coll[b] and ref[ep[b]][0] * vol1[b] / ref[ep[b]][1] < TARGET) or (not coll[b] and a['msj'][b] < TARGET)]
    log.append(f'after repair: full hexes with negative corner {int((a["msj"] < 0).sum())}; repaired elements still below {TARGET}: {still or "none"}; '
               f'worst scaled Jacobian {a["worst"]:.3f}; elements moved: {int((np.linalg.norm(P - P0, axis=1) > 0).sum())} nodes')
    print('\n'.join(log)); open(DST.replace('.vtu', '_log.txt'), 'w').write('\n'.join(log))


if __name__ == '__main__':
    main()
