"""Skin-only -> full FE head mesh with predicted skull/brain and per-node uncertainty.

    python predict_new_subject.py case_02

Dry run on a held-out HaN-Seg case: only its skin-only registration (hanseg_template_registration.py --skin-only,
which never sees skull/brain) is used as input, and the case is removed from the training set. For a real SAM3D skin,
replace `P` (the FE-topology skin from the skin-only registration) and drop the GT scoring at the end.
"""
import contextlib, io, json, os, sys
import numpy as np, pyvista as pv
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import hanseg_template_registration as h
import shape_predict as sp
from build_shape_dataset import kabsch
import k_mesh_qa
import mesh_qa_jacobian as Q

h.FE_VTU = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/mesh_model/Head_V6_repaired.vtu'   # template with its 6 real defects repaired

BASE = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/outputs'
N_CTRL = dict(skin=1500, skull=2500, brain=2500, mandible=800)
MIN_MARGIN_MM = 2.0
EXCLUDE_TRAIN = ('case_09', 'case_04')


def outward_normals(pts, faces):
    m = pv.PolyData(pts, faces).compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=False)
    n = np.asarray(m.point_normals); c = pts.mean(0)
    flip = ((pts - c) * n).sum(1) < 0
    return np.where(flip[:, None], -n, n)


def clearance(brain, skull, faces):
    """Signed distance of brain nodes to the skull surface along the nearest skull normal (< 0 inside)."""
    n = outward_normals(skull, faces); j = cKDTree(skull).query(brain)[1]
    return ((brain - skull[j]) * n[j]).sum(1), n[j]


def cavity_constraint(brain, skull, faces, margin, check_mask=None):
    """Push brain nodes that are outside (or within `margin` of) the skull surface back inside, along the skull normal.
    check_mask limits it to nodes the TEMPLATE keeps inside its skull: at the skull base the external shell is open
    and the template's own brain nodes there test as 'outside' (9.9% of them), so they are left alone."""
    s, n_j = clearance(brain, skull, faces); n = n_j
    bad = (s > -margin) if check_mask is None else (s > -margin) & check_mask
    j = np.arange(len(brain))
    out = brain.copy(); out[bad] -= ((s[bad] + margin)[:, None]) * n[bad]
    return out, int(bad.sum())


def mesh_qa(vtu_points, template):
    nid = np.asarray(template.point_data['node_id']); nodes = {int(i): p for i, p in zip(nid, vtu_points)}
    conn, off, ct = template.cell_connectivity, template.offset, template.celltypes
    eid, pid = np.asarray(template.cell_data['element_id']), np.asarray(template.cell_data['part_id'])
    solid, shell = {}, {}
    for c in range(template.n_cells):
        ids = [int(nid[k]) for k in conn[off[c]:off[c + 1]]]
        (solid if ct[c] == 12 else shell)[int(eid[c])] = (int(pid[c]), ids)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = k_mesh_qa.check_mesh(nodes, solid, label='', elems_shell=shell)
    return r, buf.getvalue()


def main(case, smooth=1000.0, tag=''):
    out = f'{BASE}/predict_demo/{case}{tag}'; os.makedirs(out, exist_ok=True); log = []
    z = np.load(f'{sp.DS}/shape_dataset.npz'); cases = [str(c) for c in z['cases']]
    sr, kr, br, mr = z['skin_reliable'], z['skull_reliable'], z['brain_reliable'], z['mandible_reliable']
    tr = [i for i, c in enumerate(cases) if c != case and c not in EXCLUDE_TRAIN]
    skin_tr = z['skin'][tr][:, sr]; y_tr = np.concatenate([z['skull'][tr][:, kr], z['brain'][tr][:, br]], axis=1)
    ns = int(kr.sum())

    # 1. input: FE-topology skin from the skin-only registration (no skull/brain of this subject is used)
    DEMO = os.environ.get('DEMO_SKIN_DIR')   # real input with no ground truth: skin-only registration folder, GT check skipped
    P = np.asarray(pv.read(f'{DEMO or BASE + "/hanseg_template_reg_skinonly"}/{case}/{case}_skin_registered.vtp').points)
    R, t = kabsch(P[sr], skin_tr.mean(0)); skin_q = P[sr] @ R.T + t           # aligned = raw @ R.T + t
    to_raw = lambda x: (x - t) @ R

    # 2. predict with case-specific uncertainty
    m = sp.pca_ridge_fit(skin_tr, y_tr); calib = sp.fit_calibration(skin_tr, y_tr)
    pred, sigma = sp.pca_ridge_predict(m, skin_q, calib); pred = to_raw(pred)
    skull_p, brain_p = pred[:ns], pred[ns:]
    # mandible: its own PCA-ridge model on the same skin input, own calibration
    y_m = z['mandible'][tr][:, mr]; mm = sp.pca_ridge_fit(skin_tr, y_m); calib_m = sp.fit_calibration(skin_tr, y_m)
    mand_p, sigma_m = sp.pca_ridge_predict(mm, skin_q, calib_m); mand_p = to_raw(mand_p)
    log.append(f'mandible: calibration a={calib_m[0]:.2f} b={calib_m[1]:.2f}; sigma mean {sigma_m.mean():.2f} mm')
    log.append(f'calibration a={calib[0]:.2f} b={calib[1]:.2f}; sigma mean {sigma.mean():.2f} mm '
               f'(skull {sigma[:ns].mean():.2f}, brain {sigma[ns:].mean():.2f}); 95% radius = {np.sqrt(sp.CHI95):.2f} x sigma')

    # 3. cavity constraint on the predicted brain against the predicted skull surface
    fe, names, pid = h.load_fe(); S, _ = h.fe_structures(fe, names, pid)
    T = {k: np.asarray(v.points, float) for k, v in S.items()}
    faces = S['skull'].faces
    skull_t, brain_t = T['skull'][kr], T['brain'][br]
    n_t = outward_normals(T['skull'], faces); j_t = cKDTree(T['skull']).query(T['brain'][br])[1]
    s_t = ((T['brain'][br] - T['skull'][j_t]) * n_t[j_t]).sum(1)              # template clearance, < 0 inside
    margin = max(MIN_MARGIN_MM, 0.5 * float(-np.percentile(s_t, 95)))
    skull_full = skull_p                                                       # kr is all-true
    enclosed_t = s_t <= -margin                                                # nodes the template itself keeps inside
    brain_c, n_bad = cavity_constraint(brain_p, skull_full, faces, margin, check_mask=enclosed_t)
    log.append(f'cavity constraint (on the {int(enclosed_t.sum())} nodes the template keeps inside; {int((~enclosed_t).sum())} skull-base nodes skipped): '
               f'margin {margin:.1f} mm; {n_bad} predicted brain nodes were outside it and were moved '
               f'(max move {np.linalg.norm(brain_c - brain_p, axis=1).max():.1f} mm)')

    # 4. propagate to the whole FE mesh: TPS from template surface nodes to their (registered / predicted) positions
    rng = np.random.default_rng(0); pick = lambda n, k: rng.choice(n, min(k, n), replace=False)
    si, ki, bi = np.where(sr)[0], np.arange(ns), np.where(br)[0]
    ps, pk, pb = pick(len(si), N_CTRL['skin']), pick(len(ki), N_CTRL['skull']), pick(len(bi), N_CTRL['brain'])
    mand_t = T['mandible'][mr]; pm = pick(len(mand_t), N_CTRL['mandible'])
    src = np.vstack([T['skin'][si[ps]], skull_t[pk], brain_t[pb], mand_t[pm]]); dst = np.vstack([P[si[ps]], skull_full[pk], brain_c[pb], mand_p[pm]])
    tps = RBFInterpolator(src, dst, kernel='thin_plate_spline', smoothing=smooth)
    X = np.asarray(fe.points, float); new = np.vstack([tps(X[a:a + 20000]) for a in range(0, len(X), 20000)])

    # 5. uncertainty field on every node: sigma of the nearest predicted skull/brain node
    sig_pts = np.vstack([skull_t, brain_t, mand_t]); node_sigma = np.concatenate([sigma, sigma_m])[cKDTree(sig_pts).query(X)[1]]
    res = fe.copy(); res.points = new
    res.point_data['sigma_mm'] = node_sigma; res.point_data['radius95_mm'] = np.sqrt(sp.CHI95) * node_sigma
    res.save(f'{out}/{case}_predicted_head.vtu')
    log.append(f'TPS smoothing {smooth:g}')

    if not DEMO:
        # 6. accuracy against this case's GT (surface nodes of the final full mesh) -- dry run only
        nid2idx = {int(v): i for i, v in enumerate(fe.point_data['node_id'])}
        got = lambda k, mask: new[[nid2idx[int(v)] for v in np.asarray(S[k].point_data['node_id'])[mask]]]
        tg = lambda k: np.load(f'{BASE}/hanseg_template_reg/{case}/{case}_{k}_target_trimmed.npy')
        sc = {'skull': got('skull', kr), 'brain': got('brain', br), 'mandible': got('mandible', mr)}
        from validate_shape_predictors import chamfer_hd95
        for k in sc:
            c, p95 = chamfer_hd95(sc[k], tg(k)); log.append(f'GT check [{k}] full-mesh surface nodes: Chamfer {c:.2f} mm, HD95 {p95:.1f} mm')
        s_a = np.linalg.norm(np.vstack([sc['skull'], sc['brain'], sc['mandible']]) - np.vstack([skull_full[:, :], brain_c, mand_p]), axis=1)
        log.append(f'TPS reproduces the predicted surfaces: median shift {np.median(s_a):.2f} mm, max {s_a.max():.1f} mm')

    # 7. mesh QA with the Jacobian-based checker (k_mesh_qa's majority-vote proxy mis-flags collapsed hexes; kept as 'legacy')
    a = Q.assess(fe, new, template_pts=np.asarray(fe.points, float))
    log.append(f'inverted hexes (min scaled Jacobian < 0, collapsed by sign flip): {a["n_bad"]} of {a["n_hex"]} '
               f'(template {a["n_bad_template"]}); worst scaled Jacobian {a["worst"]:.3f}; elements below 0.05: {a["n_below_0_05"]}; '
               f'worst parts {[(names.get(int(p), p), n) for p, n in Q.by_part(fe, a, 4)]}')
    r1, txt = mesh_qa(new, fe); r0, _ = mesh_qa(np.asarray(fe.points, float), fe)
    log.append(f'legacy k_mesh_qa flags: template {r0["n_inverted"]}, propagated {r1["n_inverted"]} (not reliable)')
    open(f'{out}/{case}_qa.txt', 'w').write(txt); open(f'{out}/{case}_log.txt', 'w').write('\n'.join(log)); print('\n'.join(log))


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    sm = float(next((x.split('=')[1] for x in sys.argv if x.startswith('--smooth=')), 1000.0))
    main(a[0] if a else 'case_02', smooth=sm, tag=f'_smooth{sm:g}' if sm != 1000.0 else '')
