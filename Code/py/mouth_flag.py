"""Mouth-misfit flag from the input skin and the registered skin only (no internal ground truth).

Band: midline +-30 mm, 12-60 mm below the input nose tip, anterior of the nose tip - 30 mm. For every input skin point in the band the true
distance to the registered skin surface is taken; the flag is raised when the 95th percentile exceeds P95_FLAG_MM (9 mm; the other cases sit at 2-7 mm). A flagged mouth means the registered lips/chin region should not be trusted. The flag does not say why: open mouths are one cause,
and the 39 cases flag 12, 13, 23 and 34 (cases 16 and 32 look open but stay below the threshold).
"""
import numpy as np
from nose_landmark import find_nose

P95_FLAG_MM = 9.0


def mouth_misfit(reg_polydata, gt_skin):
    G = np.asarray(gt_skin, float); top = G[:, 2].max(); dep = top - G[:, 2]
    nt = G[find_nose(G)]; nd = top - nt[2]; xm = 0.5 * (G[G[:, 2] > top - 100, 0].min() + G[G[:, 2] > top - 100, 0].max())
    m = (np.abs(G[:, 0] - xm) < 30) & (dep > nd + 12) & (dep < nd + 60) & (G[:, 1] < nt[1] + 30)
    if m.sum() < 50: return dict(p95_mm=float('nan'), max_mm=float('nan'), flag=None, n=int(m.sum()))
    cp = reg_polydata.find_closest_cell(G[m], return_closest_point=True)[1]; d = np.linalg.norm(G[m] - cp, axis=1)
    p95 = float(np.percentile(d, 95))
    return dict(p95_mm=p95, max_mm=float(d.max()), flag=bool(p95 > P95_FLAG_MM), n=int(m.sum()))

