"""Pictures + Slicer-ready surfaces for the similarity.vtp demo (skin-only input -> registered skin -> predicted skull/brain/mandible -> FE mesh)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyvista as pv
import hanseg_template_registration as h
B = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data'; O = f'{B}/outputs/predict_demo/similarity'
h.FE_VTU = f'{B}/mesh_model/Head_V6_repaired.vtu'
fe, names, pid = h.load_fe(); S, _ = h.fe_structures(fe, names, pid)
fin = pv.read(f'{O}/similarity_predicted_head_untangled_tau0.2.vtu'); P = np.asarray(fin.points)
nid = {int(v): i for i, v in enumerate(fe.point_data['node_id'])}; sig = np.asarray(fin.point_data['sigma_mm'])
os.makedirs(f'{O}/slicer', exist_ok=True)
for k in ['skull', 'brain', 'mandible']:
    s = S[k].copy(); idx = [nid[int(v)] for v in np.asarray(S[k].point_data['node_id'])]; s.points = P[idx]; s.point_data['sigma_mm'] = sig[idx]
    s.save(f'{O}/slicer/similarity_pred_{k}.vtp')
skin = pv.read(f'{B}/outputs/hanseg_template_reg_skinonly_demo/similarity/similarity_skin_registered.vtp'); skin.save(f'{O}/slicer/similarity_registered_skin.vtp')
inp = pv.read('/Users/lapastang/Downloads/similarity.vtp'); inp.save(f'{O}/slicer/similarity_input_skin.vtp')
fin.save(f'{O}/slicer/similarity_FE_head_untangled.vtu')
pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2400, 800))
def cam(p): p.camera_position = 'yz'; p.camera.azimuth = 0; p.camera.SetViewUp(0, 0, 1); p.camera.position = (-700, 40, -800 + 0) if False else p.camera.position
pl.subplot(0, 0); pl.add_mesh(inp, color='lightgray', opacity=0.6); pl.add_mesh(skin, color='orange', style='wireframe', line_width=1); pl.add_text('input skin (grey) + registered FE skin (orange)', font_size=9)
pl.subplot(0, 1); pl.add_mesh(skin, color='tan', opacity=0.18)
for k, c in [('skull', 'ivory'), ('brain', 'salmon'), ('mandible', 'lightblue')]: pl.add_mesh(pv.read(f'{O}/slicer/similarity_pred_{k}.vtp'), color=c, opacity=0.9 if k != 'skull' else 0.45)
pl.add_text('predicted skull / brain / mandible', font_size=9)
pl.subplot(0, 2); half = fin.clip(normal=(1, 0, 0), origin=(float(P[:, 0].mean()), 0, 0), invert=False)
pl.add_mesh(half, scalars='sigma_mm', cmap='viridis', clim=(0, 8), show_edges=False, scalar_bar_args=dict(title='sigma mm')); pl.add_text('FE mesh, half cut, coloured by uncertainty', font_size=9)
for i in range(3):
    pl.subplot(0, i); pl.view_vector((-1, 0, 0)); pl.set_viewup((0, 0, 1))
pl.screenshot('/tmp/halfcut/similarity_demo.png'); print('saved')
