"""Skull/brain surface error (vs this case's GT target points) of a predicted mesh before and after untangling."""
import os, sys, numpy as np, pyvista as pv
sys.path.insert(0, '.')
import hanseg_template_registration as h
h.FE_VTU = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/mesh_model/Head_V6_repaired.vtu'
import shape_predict as sp
from validate_shape_predictors import chamfer_hd95
BASE = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/outputs'
z = np.load(f'{sp.DS}/shape_dataset.npz'); kr, br = z['skull_reliable'], z['brain_reliable']
fe, names, pid = h.load_fe(); S, _ = h.fe_structures(fe, names, pid)
nid2idx = {int(v): i for i, v in enumerate(fe.point_data['node_id'])}
idx = {k: np.array([nid2idx[int(v)] for v in np.asarray(S[k].point_data['node_id'])[m]]) for k, m in (('skull', kr), ('brain', br))}
for tag in sys.argv[1:]:
    case = tag.split('_smooth')[0]
    tg = lambda k: np.load(f'{BASE}/hanseg_template_reg/{case}/{case}_{k}_target_trimmed.npy')
    row = []
    for suffix in ('', os.environ.get('SUFFIX', '_untangled')):
        P = np.asarray(pv.read(f'{BASE}/predict_demo/{tag}/{case}_predicted_head{suffix}.vtu').points)
        row.append({k: chamfer_hd95(P[idx[k]], tg(k)) for k in idx})
    for k in idx:
        (c0, p0), (c1, p1) = row[0][k], row[1][k]
        print(f'{tag:20s} {k:5s} Chamfer {c0:.2f} -> {c1:.2f} mm ({c1-c0:+.2f}); HD95 {p0:.1f} -> {p1:.1f} mm')
