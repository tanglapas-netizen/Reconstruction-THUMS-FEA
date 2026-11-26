"""Mouth-misfit flag (mouth-interior points are removed first, see mouth_interior_mask) from the input skin and the registered skin only (no internal ground truth).

Band: midline +-30 mm, 12-60 mm below the input nose tip, anterior of the nose tip - 30 mm. For every input skin point in the band the true
distance to the registered skin surface is taken; the flag is raised when the 95th percentile exceeds P95_FLAG_MM (7.5 mm, chosen by eye from the 39 cases: after interior removal 13 and 23 sit at 7.8-8.0, all others at <=6.7). A flagged mouth means the registered lips/chin region should not be trusted. The flag does not say why: open mouths are one cause,
and the 39 cases flag 12, 13, 23 and 34 (cases 16 and 32 look open but stay below the threshold).
"""
import numpy as np
from nose_landmark import find_nose

P95_FLAG_MM = 7.5


def mouth_misfit(reg_polydata, gt_skin):
    G = np.asarray(gt_skin, float); top = G[:, 2].max(); dep = top - G[:, 2]
    nt = G[find_nose(G)]; nd = top - nt[2]; xm = 0.5 * (G[G[:, 2] > top - 100, 0].min() + G[G[:, 2] > top - 100, 0].max())
    m = (np.abs(G[:, 0] - xm) < 30) & (dep > nd + 12) & (dep < nd + 60) & (G[:, 1] < nt[1] + 30)
    m &= ~mouth_interior_mask(G)   # score the lips and chin only, not the inside of the mouth
    if m.sum() < 50: return dict(p95_mm=float('nan'), max_mm=float('nan'), flag=None, n=int(m.sum()))
    cp = reg_polydata.find_closest_cell(G[m], return_closest_point=True)[1]; d = np.linalg.norm(G[m] - cp, axis=1)
    p95 = float(np.percentile(d, 95))
    return dict(p95_mm=p95, max_mm=float(d.max()), flag=bool(p95 > P95_FLAG_MM), n=int(m.sum()))


MOUTH_DEPTH_MM = (5.0, 75.0)   # below the nose tip, where the interior can appear
BEHIND_MM = 8.0                # a point this far behind the front surface in its own line of sight is interior
CELL_MM = 3.0


def mouth_interior_mask(gt_skin, half_width=40.0):
    """True for input skin points that lie inside the mouth: in the mouth band, more than BEHIND_MM behind the frontmost skin surface seen
    along the same front-to-back line (cells of CELL_MM in x,z). The lips are the front surface, so only points deeper in the mouth are marked."""
    G = np.asarray(gt_skin, float); top = G[:, 2].max(); dep = top - G[:, 2]
    nt = G[find_nose(G)]; nd = top - nt[2]; xm = 0.5 * (G[G[:, 2] > top - 100, 0].min() + G[G[:, 2] > top - 100, 0].max())
    band = (np.abs(G[:, 0] - xm) < half_width) & (dep > nd + MOUTH_DEPTH_MM[0]) & (dep < nd + MOUTH_DEPTH_MM[1]) & (G[:, 1] < nt[1] + 40)
    idx = np.where(band)[0]; key = np.round(G[idx][:, [0, 2]] / CELL_MM).astype(int)
    front = {}
    for k, y in zip(map(tuple, key), G[idx, 1]):
        if k not in front or y < front[k]: front[k] = y
    out = np.zeros(len(G), bool)
    for i, k in zip(idx, map(tuple, key)): out[i] = G[i, 1] > front[k] + BEHIND_MM
    return out
