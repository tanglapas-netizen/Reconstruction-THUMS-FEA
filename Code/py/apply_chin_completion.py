"""Apply chin completion to the saved 221-cut skin-only registrations (equivalent to re-running hanseg_template_registration.py --skin-only
with chin completion, without repeating the 1-minute fits). Reads hanseg_template_reg_skinonly_cut221/<case>, writes hanseg_template_reg_skinonly/<case>."""
import glob, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree
from chin_complete import complete_chin
B = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data'
SRC, DST = f'{B}/outputs/hanseg_template_reg_skinonly_cut221', f'{B}/outputs/hanseg_template_reg_skinonly'
EXCL = {'case_09'}
for d in sorted(glob.glob(f'{SRC}/case_*')):
    c = os.path.basename(d)
    if c in EXCL: continue
    os.makedirs(f'{DST}/{c}', exist_ok=True)
    r = pv.read(f'{d}/{c}_skin_registered.vtp'); gt = np.asarray(pv.read(f'{B}/HaN-Seg/GT/usable/{c}/stl/{c}_skin_surface.stl').points, float)
    P = np.asarray(r.points); act = np.asarray(r.point_data['active']).astype(bool); Q = complete_chin(P, act, gt)
    near = cKDTree(gt).query(Q)[0] < 8.0
    r.points = Q; r.point_data['active_completed'] = (act | near).astype(np.uint8); r.save(f'{DST}/{c}/{c}_skin_registered.vtp')
    for f in glob.glob(f'{d}/{c}_*'):
        if not f.endswith('_skin_registered.vtp'): shutil.copy(f, f'{DST}/{c}/')
    top = gt[:, 2].max(); dd = cKDTree(gt).query(Q)[0]; ch = (top - Q[:, 2] > 180) & (Q[:, 1] < 30)
    print(f'{c}: chin/jaw mean {dd[ch].mean():.1f} max {dd[ch].max():.1f}; fitted nodes {int(act.sum())} -> {int((act | near).sum())}')
