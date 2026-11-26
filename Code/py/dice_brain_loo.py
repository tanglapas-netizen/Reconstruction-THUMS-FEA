"""Leave-one-out Dice of the predicted brain (skin-only -> PCA-ridge) against the CT brain segmentation (HaN-Seg GT STL), 2 mm voxels."""
import sys, os, json, numpy as np, pyvista as pv
import hanseg_template_registration as h, shape_predict as sp
from build_shape_dataset import kabsch
h.FE_VTU = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/mesh_model/Head_V6_repaired.vtu'
B = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data'
z = np.load(f'{sp.DS}/shape_dataset.npz'); cases = [str(c) for c in z['cases']]; sr, kr, br = z['skin_reliable'], z['skull_reliable'], z['brain_reliable']
fe, names, pid = h.load_fe(); S, _ = h.fe_structures(fe, names, pid); bs = S['brain']; print('brain reliable', br.sum(), 'of', len(br))
def mask(surf, grid):
    s = pv.PolyData(np.asarray(surf.points), surf.faces).clean(); return np.asarray(grid.select_enclosed_points(s, tolerance=0.0, check_surface=False)['SelectedPoints']).astype(bool)
EX = ('case_09', 'case_04'); res = {}
for i, c in enumerate(cases):
    if c in EX: continue
    tr = [j for j, k in enumerate(cases) if k != c and k not in EX]
    skin_tr = z['skin'][tr][:, sr]; y_tr = np.concatenate([z['skull'][tr][:, kr], z['brain'][tr][:, br]], axis=1); ns = int(kr.sum())
    P = z['skin'][i][sr]; R, t = kabsch(P, skin_tr.mean(0)); q = P @ R.T + t
    m = sp.pca_ridge_fit(skin_tr, y_tr); pred, _ = sp.pca_ridge_predict(m, q, sp.fit_calibration(skin_tr, y_tr)); pred = (pred - t) @ R
    if os.environ.get('BASELINE') == 'mean': pred = (np.tile(y_tr.mean(0), (1, 1))[0] - t) @ R if False else (y_tr.mean(0).reshape(-1, 3) - t) @ R
    bp = pred[ns:]
    Tb = np.asarray(bs.points, float); full = Tb.copy(); full[br] = bp
    if not br.all():   # nodes with no reliable ground truth: carry them with a TPS of the reliable predicted nodes
        from scipy.interpolate import RBFInterpolator
        ci = np.where(br)[0][::6]; full[~br] = RBFInterpolator(Tb[ci], bp[np.searchsorted(np.where(br)[0], ci)], kernel='thin_plate_spline', smoothing=1.0)(Tb[~br])
    pb = bs.copy(); pb.points = full; bp = full
    # GT brain STL is in the scanner frame; bring it into the dataset frame through the skin (raw registered skin -> dataset skin)
    raw = np.asarray(pv.read(f'{B}/outputs/hanseg_template_reg_skinonly/{c}/{c}_skin_registered.vtp').points)[sr]
    R0, t0 = kabsch(raw, P); resid = np.linalg.norm(raw @ R0.T + t0 - P, axis=1).mean()
    g = pv.read(f'{B}/HaN-Seg/GT/usable/{c}/stl/{c}_brain.stl'); g.points = np.asarray(g.points) @ R0.T + t0
    lo = np.minimum(g.points.min(0), bp.min(0)) - 5; hi = np.maximum(g.points.max(0), bp.max(0)) + 5
    dims = np.ceil((hi - lo) / 2.0).astype(int) + 1; grid = pv.ImageData(dimensions=dims, spacing=(2, 2, 2), origin=lo)
    a, b = mask(pb, grid), mask(g, grid); d = 2 * (a & b).sum() / (a.sum() + b.sum()); res[c] = float(d)
    # oracle: template-registered brain (fit to the true GT brain)
    print(c, 'frame resid %.2f' % resid, 'dice %.3f' % d, 'vol pred/gt %.0f/%.0f mL' % (a.sum() * 8e-3, b.sum() * 8e-3), flush=True)
v = np.array(list(res.values())); print('LOO brain Dice: mean %.3f median %.3f min %.3f max %.3f n %d' % (v.mean(), np.median(v), v.min(), v.max(), len(v)))
json.dump(res, open('/private/tmp/claude-501/-Users-lapastang-Desktop-Reconstruction-THUMS-FEA/c414ab5b-e36d-4757-b96b-7124e040cadd/scratchpad/dice_brain.json', 'w'))
