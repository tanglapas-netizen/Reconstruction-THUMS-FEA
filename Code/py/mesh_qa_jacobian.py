"""Hex-mesh inversion check that does not mis-flag collapsed (tet / wedge-as-hex) elements.

k_mesh_qa.check_mesh decides 'inverted' by a per-part majority vote on a centroid-based volume proxy. On Head_V6 that flags
7,121 hexes, but 4,113 are collapsed hexes (4 or 6 distinct nodes, valid by design) and 3,008 have a positive Jacobian at all
8 corners; only 2 elements actually have a negative corner. This module judges elements directly:
  full hex (8 distinct nodes) : minimum scaled Jacobian over the 8 corners (negative = inverted)
  collapsed hex               : sign of the centroid volume proxy compared with the TEMPLATE's own sign (flip = inverted by the warp)
"""
import collections
import numpy as np

NB = {0: (1, 3, 4), 1: (2, 0, 5), 2: (3, 1, 6), 3: (0, 2, 7), 4: (7, 5, 0), 5: (4, 6, 1), 6: (5, 7, 2), 7: (6, 4, 3)}
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]


def hex_connectivity(mesh):
    hx = np.where(mesh.celltypes == 12)[0]; conn, off = mesh.cell_connectivity, mesh.offset
    C = np.stack([conn[off[c]:off[c] + 8] for c in hx]); coll = np.array([len(set(c)) < 8 for c in C])
    return hx, C, coll


def min_scaled_jacobian(X):
    out = []
    for i in range(8):
        e = [X[:, a] - X[:, i] for a in NB[i]]
        L = np.prod([np.linalg.norm(v, axis=1) for v in e], axis=0)
        out.append(np.linalg.det(np.stack(e, axis=1)) / np.maximum(L, 1e-12))
    return np.stack(out, axis=1).min(1)


def proxy_volume(X):
    c = X.mean(1); v = np.zeros(len(X))
    for f in FACES:
        a, b, cc, d = [X[:, i] - c for i in f]
        v += (np.einsum('ij,ij->i', a, np.cross(b, cc)) + np.einsum('ij,ij->i', a, np.cross(cc, d))) / 6.0
    return v


def valid_corner_sj_min(C, X):
    """Minimum scaled Jacobian over the corners whose 3 neighbours are distinct nodes (the corners of a collapsed hex that are still hexahedral).
    C: (n,8) node ids, X: (n,8,3) coordinates. Template wedges/tets are positive at every such corner, so <= 0 means inverted."""
    out = np.full((len(C), 8), np.inf)
    for i in range(8):
        a, b, d = NB[i]
        ok = np.array([len({r[0], r[1], r[2], r[3]}) == 4 for r in zip(C[:, i], C[:, a], C[:, b], C[:, d])])
        e = [X[:, k] - X[:, i] for k in (a, b, d)]
        L = np.prod([np.linalg.norm(v, axis=1) for v in e], axis=0)
        out[:, i] = np.where(ok, np.linalg.det(np.stack(e, axis=1)) / np.maximum(L, 1e-12), np.inf)
    return out.min(1)


def assess(mesh, pts, template_pts=None, tol=0.0):
    """Returns per-hex arrays and a summary. template_pts defaults to the mesh's own points."""
    P0 = np.asarray(mesh.points, float) if template_pts is None else template_pts
    hx, C, coll = hex_connectivity(mesh); X0, X1 = P0[C], np.asarray(pts, float)[C]
    msj = np.where(coll, np.nan, 0.0); msj[~coll] = min_scaled_jacobian(X1[~coll])
    # collapsed hexes are judged on their still-hexahedral corners (the old centroid-proxy sign test gave false positives on 6-node wedges)
    cs1 = valid_corner_sj_min(C[coll], X1[coll]); cs0 = valid_corner_sj_min(C[coll], X0[coll])
    bad = msj < tol; bad[coll] = cs1 <= tol
    msj0 = np.where(coll, np.nan, 0.0); msj0[~coll] = min_scaled_jacobian(X0[~coll])
    bad0 = msj0 < tol; bad0[coll] = cs0 <= tol
    return dict(hx=hx, coll=coll, msj=msj, bad=bad, bad0=bad0,
                n_hex=len(hx), n_collapsed=int(coll.sum()), n_bad=int(bad.sum()), n_bad_template=int(bad0.sum()),
                n_new=int((bad & ~bad0).sum()), n_fixed=int((bad0 & ~bad).sum()),
                worst=float(np.nanmin(msj)), n_below_0_05=int((msj < 0.05).sum()))


def by_part(mesh, r, top=6):
    ep = np.asarray(mesh.cell_data['part_id'])[r['hx']]
    return collections.Counter(ep[r['bad']].tolist()).most_common(top)
