"""Local untangling of a warped hex mesh (port of MaiAlgo notebook section 15, cells 31-33, to our pipeline).

Nodes of hexes whose corner scaled Jacobian is < tau, plus RINGS rings of neighbours, are free. L-BFGS-B minimises
  sum over corners with SJ < tau of (tau - SJ)^2  +  mu * |x - x_pred|^2   (mu x SKIN_X on the skin)
with analytic gradient and a move cap (BOX = interior mm, skin mm). Collapsed hexes: only corners whose three neighbours are
distinct nodes are used (the same 'valid' rule as MaiAlgo).
Usage: python untangle_mesh.py <predicted_head.vtu> [--template Head_V6_repaired.vtu] [--tau 0.1] [--out file.vtu]
"""
import argparse, os, sys, time
import numpy as np, pyvista as pv
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_qa_jacobian as Q

SKIN_IDS = np.array(list(range(88000167, 88000173)) + list(range(88000221, 88000237)))   # face_skin_*/head_skin_* parts (the old 160-240 range also covered muscles and eye parts)
MU, SKIN_X, BOX_INT, BOX_SKIN, RINGS, TAU = 1e-4, 20.0, 5.0, 1.5, 1, 0.10
TEMPLATE = "/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/mesh_model/Head_V6_repaired.vtu"


class HexSet:
    def __init__(self, mesh):
        self.hx, self.C, self.coll = Q.hex_connectivity(mesh)
        C = self.C
        self.valid = np.zeros((len(C), 8), bool)                       # corner usable (3 distinct neighbours, distinct from itself)
        for c in range(8):
            a, b, d = Q.NB[c]
            n = [C[:, c], C[:, a], C[:, b], C[:, d]]
            self.valid[:, c] = [len({*r}) == 4 for r in zip(*n)]
        pid = np.asarray(mesh.cell_data["part_id"])[self.hx]
        sk = np.isin(pid, SKIN_IDS)
        self.skin_node = np.zeros(mesh.n_points, bool); self.skin_node[np.unique(C[sk])] = True

    def corner_sj(self, X, ce=None):
        ce = np.arange(len(self.C)) if ce is None else ce
        P = X[self.C[ce]]; out = np.full((len(ce), 8), np.nan)
        for c in range(8):
            a, b, d = Q.NB[c]
            e1, e2, e3 = P[:, a] - P[:, c], P[:, b] - P[:, c], P[:, d] - P[:, c]
            n = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)
            s = np.einsum("ij,ij->i", e1, np.cross(e2, e3)) / np.maximum(n, 1e-300)
            out[:, c] = np.where(self.valid[ce, c], s, np.nan)
        return out

    def energy(self, X, ce, tau):
        P = X[self.C[ce]]; E, G = 0.0, np.zeros_like(X)
        for c in range(8):
            a, b, d = Q.NB[c]
            e1, e2, e3 = P[:, a] - P[:, c], P[:, b] - P[:, c], P[:, d] - P[:, c]
            n1, n2, n3 = (np.linalg.norm(e, axis=1) for e in (e1, e2, e3))
            den = np.maximum(n1 * n2 * n3, 1e-300)
            c23, c31, c12 = np.cross(e2, e3), np.cross(e3, e1), np.cross(e1, e2)
            s = np.einsum("ij,ij->i", e1, c23) / den
            act = self.valid[ce, c] & (s < tau)
            if not act.any(): continue
            w = np.where(act, -2.0 * (tau - s), 0.0); E += float(np.sum(np.where(act, (tau - s) ** 2, 0.0)))
            g1 = w[:, None] * (c23 / den[:, None] - s[:, None] * e1 / np.maximum(n1 ** 2, 1e-300)[:, None])
            g2 = w[:, None] * (c31 / den[:, None] - s[:, None] * e2 / np.maximum(n2 ** 2, 1e-300)[:, None])
            g3 = w[:, None] * (c12 / den[:, None] - s[:, None] * e3 / np.maximum(n3 ** 2, 1e-300)[:, None])
            nd = self.C[ce]
            np.add.at(G, nd[:, a], g1); np.add.at(G, nd[:, b], g2); np.add.at(G, nd[:, d], g3); np.add.at(G, nd[:, c], -(g1 + g2 + g3))
        return E, G


def untangle(mesh, X, tau=TAU, mu=MU, rings=RINGS, maxiter=800, box=(BOX_INT, BOX_SKIN), log=print, H=None):
    H = H or HexSet(mesh); X0 = np.asarray(X, float)
    sj = H.corner_sj(X0); bad = np.nanmin(np.where(np.isnan(sj), 9, sj), axis=1) < tau
    free = np.zeros(len(X0), bool); free[np.unique(H.C[bad])] = True
    for _ in range(rings): free[np.unique(H.C[free[H.C].any(1)])] = True
    fi = np.where(free)[0]; ce = np.where(free[H.C].any(1))[0]
    skin = H.skin_node[fi]; mu_i = np.where(skin, mu * SKIN_X, mu)[:, None]
    lim = np.where(skin, box[1], box[0])[:, None] * np.ones((1, 3)); x0 = X0[fi]
    def fun(v):
        Xk = X0.copy(); Xk[fi] = v.reshape(-1, 3)
        E, G = H.energy(Xk, ce, tau); d = Xk[fi] - x0
        return E + float((mu_i * d ** 2).sum()), (G[fi] + 2.0 * mu_i * d).ravel()
    t = time.time()
    res = minimize(fun, x0.ravel(), jac=True, method="L-BFGS-B", bounds=list(zip((x0 - lim).ravel(), (x0 + lim).ravel())), options=dict(maxiter=maxiter, maxfun=maxiter * 2))
    Y = X0.copy(); Y[fi] = res.x.reshape(-1, 3)
    log(f"untangle: {len(fi):,} free nodes, {len(ce):,} elements, {res.nit} iterations, {time.time()-t:.0f} s, max move {np.linalg.norm(Y[fi]-x0,axis=1).max():.2f} mm")
    return Y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("vtu"); ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--tau", type=float, default=TAU); ap.add_argument("--out"); A = ap.parse_args()
    mesh = pv.read(A.vtu); tmpl = pv.read(A.template); X = np.asarray(mesh.points, float); T0 = np.asarray(tmpl.points, float)
    H = HexSet(mesh)
    def report(tag, P):
        r = Q.assess(mesh, P, template_pts=T0); sj = np.nanmin(H.corner_sj(P), axis=1)
        print(f"{tag}: inverted {r['n_bad']} (template {r['n_bad_template']}), new {r['n_new']}, worst scaled J {r['worst']:.3f}, "
              f"below 0.05: {r['n_below_0_05']}, corner-SJ <=0: {int((sj<=0).sum())}, <0.1: {int((sj<0.1).sum())}")
        return r
    r0 = report("before", X)
    Y = untangle(mesh, X, tau=A.tau); r1 = report("after ", Y)
    d = np.linalg.norm(Y - X, axis=1); print(f"nodes moved >0.01 mm: {(d>0.01).sum():,}; max {d.max():.2f} mm; skin nodes max {d[H.skin_node].max():.2f} mm")
    out = A.out or A.vtu.replace(".vtu", "_untangled.vtu"); m2 = mesh.copy(); m2.points = Y
    m2.point_data["untangle_move_mm"] = d; m2.save(out); print("saved", out)
