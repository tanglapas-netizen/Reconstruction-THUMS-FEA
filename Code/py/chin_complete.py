"""Chin completion for the skin-only registration.

The 221 mm neck cut keeps the registration robust (a deeper cut lets long GT necks/shoulders pull the rigid CPD off in 7/39 cases), but
leaves the template chin / under-chin nodes (inactive, below the cut) unfitted and hanging 10-45 mm below the true chin. After the
registration these free nodes are pulled onto the nearest input-skin points (down to FREE_DEPTH_MM below the vertex) with a smoothed ICP
field; the active nodes, and therefore the gate metrics, do not move.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree
import hanseg_template_registration as h

FREE_DEPTH_MM, MAX_PULL_MM, ITERS, STEP, SMOOTH_K, SMOOTH_ITERS = 280.0, 60.0, 40, 0.5, 8, 2


def complete_chin(P, active, gt_skin, depth_mm=FREE_DEPTH_MM):
    top = gt_skin[:, 2].max(); g = gt_skin[gt_skin[:, 2] >= top - depth_mm]
    tree = cKDTree(g); free = ~active; p = P.copy()
    for _ in range(ITERS):
        d, nn = tree.query(p); pull = g[nn] - p
        pull[~free | (d > MAX_PULL_MM)] = 0
        for _ in range(SMOOTH_ITERS): pull = h.graph_smooth(pull, p, SMOOTH_K)
        pull[~free] = 0     # active nodes never move
        p = p + STEP * pull
    return p


if __name__ == '__main__':
    import sys
    BASE = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data'
    for c in sys.argv[1:]:
        r = pv.read(f'{BASE}/outputs/hanseg_template_reg_skinonly_cut221/{c}/{c}_skin_registered.vtp')
        gt = np.asarray(pv.read(f'{BASE}/HaN-Seg/GT/usable/{c}/stl/{c}_skin_surface.stl').points)
        P = np.asarray(r.points); act = np.asarray(r.point_data['active']).astype(bool)
        Q = complete_chin(P, act, gt); T = cKDTree(gt); top = gt[:, 2].max(); dep = top - P[:, 2]
        fl, nt = h.flipped_triangles(r, P, Q)
        for tag, X in (('221 only', P), ('completed', Q)):
            d = T.query(X)[0]; ch = (top - X[:, 2] > 180) & (X[:, 1] < 30)
            print(f'{c} {tag:9s} chin/jaw nodes mean {d[ch].mean():5.1f} max {d[ch].max():5.1f} | free nodes mean {d[~act].mean():5.1f}')
        print(f'{c} triangles reversed by completion: {fl} of {nt}')
