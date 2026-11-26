"""Apply the nose fix to saved skin-only registrations in place (equivalent to re-running --skin-only; the chin-only versions are kept in hanseg_template_reg_skinonly_chin_only)."""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyvista as pv
import hanseg_template_registration as h
from nose_landmark import find_nose, nose_fix
B = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data'
D = f'{B}/outputs/hanseg_template_reg_skinonly'
if not os.path.exists(D + '_chin_only'): shutil.copytree(D, D + '_chin_only')
fe, names, pid = h.load_fe(); S, _ = h.fe_structures(fe, names, pid); T = np.asarray(S['skin'].points, float)
for d in sorted(glob.glob(f'{D}/case_*')):
    c = os.path.basename(d); r = pv.read(f'{d}/{c}_skin_registered.vtp'); P = np.asarray(r.points)
    G = np.asarray(pv.read(f'{B}/HaN-Seg/GT/usable/{c}/stl/{c}_skin_surface.stl').points, float)
    f, _, _ = h.skin_only_alignment(T, G); a = find_nose(f(T), (100.0, 175.0)); tip = G[find_nose(G)]
    Q, dt = nose_fix(P, a, tip, G); r.points = Q; r.save(f'{d}/{c}_skin_registered.vtp'); print(f'{c}: nose node {a} moved {np.linalg.norm(dt):.1f} mm')
