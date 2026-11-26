"""Add the mouth-misfit flag to the saved skin-only results (summary.json) and print a table."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyvista as pv
from mouth_flag import mouth_misfit
B = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data'; D = f'{B}/outputs/hanseg_template_reg_skinonly'
for d in sorted(glob.glob(f'{D}/case_*')):
    c = os.path.basename(d); r = pv.read(f'{d}/{c}_skin_registered.vtp')
    G = np.asarray(pv.read(f'{B}/HaN-Seg/GT/usable/{c}/stl/{c}_skin_surface.stl').points, float)
    mf = mouth_misfit(r, G); sp = f'{d}/{c}_summary.json'; s = json.load(open(sp)); s['mouth_misfit'] = mf; json.dump(s, open(sp, 'w'), indent=1)
    print(c, 'FLAG' if mf['flag'] else 'ok  ', f"p95 {mf['p95_mm']:.1f} max {mf['max_mm']:.1f}")
