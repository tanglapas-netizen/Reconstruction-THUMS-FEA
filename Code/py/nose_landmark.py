"""Automatic nose-tip landmark on a skin surface (LPS mm, +z up, -y anterior) and a local nose correction for the skin registration.

find_nose: most anterior point of the midline face strip inside a depth window below the vertex (input skin: 70-160 mm; the FE
template sits lower after the skin-only alignment, 100-175 mm). nose_fix: the registered template nose-tip node is moved onto the
input nose tip; the displacement field is interpolated (thin-plate RBF) with every node farther than R_MM from the tip held at zero,
so the nose is raised/lengthened without dragging lips or forehead; a short local ICP then settles the nose nodes on the input surface.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

STRIP_HALF_MM, R_MM = 20.0, 45.0


def find_nose(P, window=(70.0, 160.0)):
    P = np.asarray(P, float); top = P[:, 2].max()
    r = P[P[:, 2] > top - 100]; xm = 0.5 * (r[:, 0].min() + r[:, 0].max())
    dep = top - P[:, 2]
    idx = np.where((np.abs(P[:, 0] - xm) < STRIP_HALF_MM) & (dep > window[0]) & (dep < window[1]))[0]
    return int(idx[P[idx, 1].argmin()])


def nose_fix(final, node, tip, gt_skin, r_mm=R_MM, icp_iters=10, icp_step=0.5, reach_mm=25.0, anchor_stride=2):
    """final: (N,3) registered skin; node: template nose-tip node index; tip: input nose tip; gt_skin: input skin points."""
    c = final[node]; d_tip = tip - c
    dist = np.linalg.norm(final - c, axis=1)
    anchors = np.where(dist > r_mm)[0][::anchor_stride]
    src = np.vstack([c[None], final[anchors]]); val = np.vstack([d_tip[None], np.zeros((len(anchors), 3))])
    field = RBFInterpolator(src, val, kernel='thin_plate_spline', smoothing=1e-6)
    inside = np.where(dist <= r_mm)[0]
    out = final.copy(); out[inside] = final[inside] + field(final[inside])
    # local ICP of the nose neighbourhood onto the input skin (cosine weight, no global smoothing)
    w = np.zeros(len(out)); w[inside] = 0.5 * (1 + np.cos(np.pi * np.clip(dist[inside] / r_mm, 0, 1)))
    g = gt_skin[np.linalg.norm(gt_skin - tip, axis=1) < r_mm + 25]; tree = cKDTree(g)
    for _ in range(icp_iters):
        dd, nn = tree.query(out[inside]); pull = g[nn] - out[inside]; pull[dd > reach_mm] = 0
        out[inside] += icp_step * w[inside, None] * pull
    return out, d_tip
