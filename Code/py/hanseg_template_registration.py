"""Register the FE template (Head_V6.vtu) onto one HaN-Seg GT case, structure by structure.

Phase 1/2 of the skin -> internal-organ prediction pipeline: for a case where skin, skull and
brain are all known, build FE-topology skin / skull / brain surfaces. Each structure is
registered to its own GT surface (CPD rigid -> affine -> deformable, TPS propagation, local
ICP, confidence blend) from a shared anatomical initial alignment.

usage: python hanseg_template_registration.py case_02
"""
import json, os, sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from pycpd import RigidRegistration, AffineRegistration, DeformableRegistration

BASE = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version'
FE_VTU = f'{BASE}/Data/mesh_model/Head_V6.vtu'
FE_META = f'{BASE}/Data/mesh_model/Head_V6_metadata.json'
GT_ROOT = os.environ.get('GT_ROOT', f'{BASE}/Data/HaN-Seg/GT/usable')
OUT_ROOT = f'{BASE}/Data/outputs/hanseg_template_reg'

N_LM = 2000
CPD_ITER, CPD_TOL = 100, 1e-6
CPD_ALPHA, CPD_BETA = 2.0, 2.0
TPS_SMOOTHING = 1e-3
AFFINE_CLAMP = (0.8, 1.25)
ICP_ITERS, ICP_MAX_CORR_MM, ICP_STEP, ICP_K = 10, 15.0, 0.5, 8
BLEND_TRANSITION_MM, CONF_K, CONF_ITERS = 10.0, 8, 3
SKULL_INCLUDE = ('frontal', 'parietal', 'temporal', 'occipital', 'sphenoid', 'ethmoid',
                 'lacrimal', 'nasal', 'zygomatic', 'maxilla')


# ---------------------------------------------------------------- FE template
def load_fe():
    fe = pv.read(FE_VTU)
    names = {int(k): v['title'] for k, v in json.load(open(FE_META))['parts'].items()}
    return fe, names, fe.cell_data['part_id']


def fe_surface(fe, pid, pids):
    idx = np.where(np.isin(pid, pids))[0]
    return fe.extract_cells(idx).extract_surface(algorithm='dataset_surface').triangulate()


def fe_structures(fe, names, pid):
    by_prefix = lambda *p: [k for k, n in names.items() if n.startswith(p)]
    skin = [k for k, n in names.items() if n.startswith(('face_skin_out', 'head_skin_out'))]
    skull = [k for k, n in names.items()
             if '_external_' in n and n.startswith(SKULL_INCLUDE) and 88000001 <= k <= 88000099]
    brain = [k for k, n in names.items() if n.startswith(('Pia_Cerebrum', 'Pia_Cerebellum', 'Pia_Sagittal'))]
    mandible = [k for k, n in names.items() if n.startswith('mandible') and 'shell' not in n and 88000001 <= k <= 88000099]   # all mandible solids -> outer envelope
    s = {'skin': fe_surface(fe, pid, skin), 'skull': fe_surface(fe, pid, skull), 'brain': fe_surface(fe, pid, brain),
         'mandible': fe_surface(fe, pid, mandible)}
    refs = {'cranium': by_prefix('frontal', 'parietal', 'temporal', 'occipital'),
            'face': by_prefix('maxilla', 'nasal', 'zygomatic', 'lacrimal'),
            'mandible': by_prefix('mandible')}
    centroid = lambda pids: fe.extract_cells(np.where(np.isin(pid, pids))[0]).points.mean(0)
    out = {k: centroid(v) for k, v in refs.items()}
    out['mandible_pts'] = fe.extract_cells(np.where(np.isin(pid, refs['mandible']))[0]).points
    return s, out


# ---------------------------------------------------------------- GT case
def load_gt(case):
    d = f'{GT_ROOT}/{case}/stl/{case}_'
    rd = lambda n: pv.read(d + n + '.stl')
    outer, face = rd('outer_table_surface'), [rd(n) for n in ('maxilla_palate_alveolar', 'zygomatic', 'facial_other_bone')]
    skull = pv.merge([outer] + face, merge_points=False)
    refs = {'cranium': rd('skull_cranium').points.mean(0), 'face': np.vstack([m.points for m in face]).mean(0),
            'mandible': rd('mandible_bone').points.mean(0), 'mandible_pts': rd('mandible_bone').points}
    return {'skin': rd('skin_surface'), 'skull': skull, 'brain': rd('brain'), 'mandible': rd('mandible_bone')}, refs


# ---------------------------------------------------------------- initial alignment
def anat_frame(ref):
    """Right-handed frame (anterior, superior, lateral) from three structure centroids."""
    s = ref['cranium'] - ref['mandible']; s /= np.linalg.norm(s)
    a = ref['face'] - ref['cranium']; a = a - (a @ s) * s; a /= np.linalg.norm(a)
    return np.column_stack([a, s, np.cross(a, s)])


def initial_alignment(fe_s, fe_ref, gt_s, gt_ref):
    R = anat_frame(gt_ref) @ anat_frame(fe_ref).T       # FE axes -> GT axes (proper rotation)
    rot = lambda p, c: (p - c) @ R.T
    c_fe = fe_s['skull'].points.mean(0)
    sk = rot(fe_s['skull'].points, c_fe)
    diag = lambda p: np.linalg.norm(p.max(0) - p.min(0))
    scale = diag(gt_s['skull'].points - gt_s['skull'].points.mean(0)) / diag(sk)
    c_gt = gt_s['skull'].points.mean(0)
    f = lambda p: rot(p, c_fe) * scale + c_gt
    return f, R, scale


def skin_only_alignment(fe_skin, gt_skin):
    """Initial pose from the skin surfaces alone: fixed axis map, scale from the width/length of the top
    TOP_MM of the head, translation from that region's xy centre and the vertex height."""
    R = R_FE_TO_LPS; rot = fe_skin @ R.T
    a = rot[rot[:, 2] > rot[:, 2].max() - TOP_MM]; b = gt_skin[gt_skin[:, 2] > gt_skin[:, 2].max() - TOP_MM]
    scale = float(np.mean(np.ptp(b[:, :2], axis=0) / np.ptp(a[:, :2], axis=0)))
    c_fe = np.array([a[:, 0].mean(), a[:, 1].mean(), rot[:, 2].max()])
    c_gt = np.array([b[:, 0].mean(), b[:, 1].mean(), gt_skin[:, 2].max()])
    return (lambda p: (p @ R.T - c_fe) * scale + c_gt), R, scale


# ---------------------------------------------------------------- registration
def fps(pts, n, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n, len(pts)); sel = np.empty(n, dtype=np.int64); sel[0] = rng.integers(len(pts))
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for k in range(1, n):
        sel[k] = d.argmax(); d = np.minimum(d, np.linalg.norm(pts - pts[sel[k]], axis=1))
    return sel


def agreement(src, tgt):
    d_st, i_st = cKDTree(tgt).query(src); d_ts, _ = cKDTree(src).query(tgt)
    return dict(mean=d_st.mean(), median=np.median(d_st), hd95=max(np.percentile(d_st, 95), np.percentile(d_ts, 95)),
                chamfer=d_st.mean() + d_ts.mean(), hausdorff=max(d_st.max(), d_ts.max()),
                worst_idx=int(d_st.argmax()), worst_d=d_st.max())


def graph_smooth(v, pts, k):
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    return v[idx].mean(axis=1)


TRIM_MM = 25.0
GATE = dict(chamfer=6.0, hd95=15.0, flip_frac=0.001)   # skin acceptance gate; Hausdorff is logged, not gated
COVER_GAP_MM, NECK_BAND_MM = 15.0, 50.0        # coverage gap threshold; band above the cut plane reported separately
# skin-only mode (deployment: no skull/mandible/brain of the subject exists). Constants measured on the 40 cases:
# FE->LPS axis map is the same 90 deg turn about z in every case (max deviation 0.29); vertex-to-neck-cut is
# 221 mm median (sd 13.5 mm) when the cut is the chin-based plane above.
R_FE_TO_LPS = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
TOP_MM = 100.0
VERTEX_TO_CUT_MM = float(os.environ.get('CUT_DEPTH', 221.0))   # skin-only neck cut, mm below vertex (CUT_DEPTH: experiment override). A deeper cut (250) fixes the chin but lets long necks/shoulders pull the rigid CPD off in 7 of 39 cases; the chin is completed after the fit instead (chin_complete.py)
FACE_DEPTH_MM = 221.0   # the skin gate scores the face/head above this depth only (the neck below the chin is not gated)
CUT_BELOW_CHIN_MM = 17.0   # flat horizontal neck cut this far below the lowest mandible point
ICP_SMOOTH_ITERS = 2


def trim_target(mv_rigid, fx, trim_mm):
    """Drop GT points with no FE counterpart (shoulders, neck bones): farther than trim_mm from the rigid-aligned FE."""
    keep = cKDTree(mv_rigid).query(fx)[0] < trim_mm
    return fx[keep], int((~keep).sum())


def fit(mv, fx, active, name, log, cut_z=None):
    """Rigid -> (trim target) -> rigid -> affine (guarded) -> deformable CPD -> TPS. CPD uses w=0: with an
    outlier weight w>0 rigid CPD shrinks the source to ~15% of its size and parks it inside the target."""
    cand = np.where(active)[0]
    mv_lm = mv[cand[fps(mv[cand], N_LM)]]
    rig = RigidRegistration(X=fx[fps(fx, N_LM)], Y=mv_lm, w=0, max_iterations=CPD_ITER, tolerance=CPD_TOL)
    rig.register()
    pre = rig.s * mv @ rig.R + rig.t
    if cut_z is not None:   # FE side of the neck cut: same physical plane as the GT cut, once rigidly aligned
        n_cut = int((active & (pre[:, 2] < cut_z)).sum()); active = active & (pre[:, 2] >= cut_z)
        log.append(f'[{name}] neck cut at z={cut_z:.1f}: {n_cut} FE points below the plane deactivated')
    fx, n_trim = trim_target(pre[active], fx, TRIM_MM)
    log.append(f'[{name}] trimmed {n_trim} GT points with no FE counterpart (> {TRIM_MM:.0f} mm); target now {len(fx)}')
    # same on the FE side: points with no GT counterpart (neck below the CT crop, internal fissure/folia
    # surfaces of the pia shells) are not used to drive or score the fit; the warp still carries them
    n_act = int(active.sum()); active = active & (cKDTree(fx).query(pre)[0] < TRIM_MM)
    log.append(f'[{name}] FE points without GT counterpart (> {TRIM_MM:.0f} mm): {n_act - int(active.sum())} of {n_act} excluded')
    cand = np.where(active)[0]; mv_lm = mv[cand[fps(mv[cand], N_LM)]]
    fx_lm = fx[fps(fx, N_LM)]
    rows = {'0 init': agreement(mv[active], fx)}

    rig = RigidRegistration(X=fx_lm, Y=mv_lm, w=0, max_iterations=CPD_ITER, tolerance=CPD_TOL)
    ty_r, (s, R, t) = rig.register()
    assert np.allclose(s * mv_lm @ R + t, ty_r, atol=1e-6), 'pycpd rigid convention changed'
    log.append(f'[{name}] rigid CPD scale {s:.3f}')
    dense_r = s * mv @ R + t; rows['2 rigid CPD'] = agreement(dense_r[active], fx)

    aff = AffineRegistration(X=fx_lm, Y=ty_r, w=0, max_iterations=CPD_ITER, tolerance=CPD_TOL)
    _, (B, tb) = aff.register()
    U, S, Vt = np.linalg.svd(B); Bc = U @ np.diag(np.clip(S, *AFFINE_CLAMP)) @ Vt
    ty_a_cand = ty_r @ Bc + tb
    hd = lambda a: cKDTree(fx_lm).query(a)[0].max()
    regressed = hd(ty_a_cand) > hd(ty_r)
    ty_a = ty_r if regressed else ty_a_cand
    dense_a = dense_r if regressed else dense_r @ Bc + tb
    log.append(f'[{name}] affine singular values raw {np.round(S, 3)}; landmark HD rigid {hd(ty_r):.1f} '
               f'affine {hd(ty_a_cand):.1f} -> {"fallback to rigid" if regressed else "keep affine"}')
    rows['3 affine CPD'] = agreement(dense_a[active], fx)

    dfm = DeformableRegistration(X=fx_lm, Y=ty_a, w=0, alpha=CPD_ALPHA, beta=CPD_BETA,
                                 max_iterations=CPD_ITER, tolerance=CPD_TOL)
    ty_d, _ = dfm.register()
    tps = RBFInterpolator(mv_lm, ty_d, kernel='thin_plate_spline', smoothing=TPS_SMOOTHING)
    warped = tps(mv); rows['5 TPS dense'] = agreement(warped[active], fx)
    return dict(mv=mv, fx=fx, active=active, warped=warped, rigid=dense_r, rows=rows)


def refine(ctx, icp_step=ICP_STEP, icp_iters=ICP_ITERS, smooth_k=ICP_K, smooth_iters=ICP_SMOOTH_ITERS):
    """Local ICP (pull field smoothed over the FE surface neighbourhood) + confidence blend."""
    mv, fx, active, warped = ctx['mv'], ctx['fx'], ctx['active'], ctx['warped']
    rows, tree = {}, cKDTree(fx)
    pts = warped.copy()
    for _ in range(icp_iters):
        d, nn = tree.query(pts); pull = fx[nn] - pts
        pull[(d >= ICP_MAX_CORR_MM) | ~active] = 0
        for _ in range(smooth_iters): pull = graph_smooth(pull, pts, smooth_k)
        pts = pts + icp_step * pull
    rows['6 ICP'] = agreement(pts[active], fx)
    # confidence = how much ICP improved on the TPS result. Blend toward the TPS position (smooth and fold-free),
    # NOT toward the unaligned starting pose: reverting some points to a pose ~16 mm away tears the mesh.
    d_t, _ = tree.query(warped); d_i, _ = tree.query(pts)
    conf = np.clip((d_t - d_i) / BLEND_TRANSITION_MM + 0.5, 0, 1)
    _, knn = cKDTree(pts).query(pts, k=CONF_K + 1)
    for _ in range(CONF_ITERS): conf = conf[knn].mean(axis=1)
    conf[~active] = 0.0   # no GT counterpart there: keep the TPS position
    final = warped + conf[:, None] * (pts - warped)
    rows['6.5 blend (final)'] = agreement(final[active], fx)
    return final, conf, rows


def flipped_triangles(surf, ref_pts, new_pts):
    """Triangles whose normal reverses relative to the rigidly-aligned mesh (local fold-over)."""
    f = surf.faces.reshape(-1, 4)[:, 1:]
    nrm = lambda p: np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    a, b = nrm(ref_pts), nrm(new_pts)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return int((cos < 0).sum()), len(f)


def main(case, skin_only=False, only=None):
    out = f"{OUT_ROOT}_skinonly{os.environ.get('OUT_TAG', '')}/{case}" if skin_only else f'{OUT_ROOT}/{case}'; os.makedirs(out, exist_ok=True); log = []
    fe, names, pid = load_fe(); fe_s, fe_ref = fe_structures(fe, names, pid)
    if skin_only:
        fe_s = {'skin': fe_s['skin']}
        gt_s = {'skin': pv.read(f'{GT_ROOT}/{case}/stl/{case}_skin_surface.stl')}
        if os.environ.get('KEEP_MOUTH_INTERIOR', '0') != '1':   # drop the inside-of-mouth surface from the target (mouth_flag.mouth_interior_mask)
            from mouth_flag import mouth_interior_mask
            _P = np.asarray(gt_s['skin'].points, float); _m = mouth_interior_mask(_P); gt_s['skin'] = pv.PolyData(_P[~_m])
            log.append(f'mouth interior removed from the input skin: {int(_m.sum())} of {len(_P)} points')
        f, R, scale = skin_only_alignment(np.asarray(fe_s['skin'].points, dtype=float), np.asarray(gt_s['skin'].points, dtype=float))
    else:
        gt_s, gt_ref = load_gt(case)
        f, R, scale = initial_alignment(fe_s, fe_ref, gt_s, gt_ref)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    log.append(f'init: rotation {ang:.1f} deg, scale {scale:.3f}')
    log.append('FE->GT axis map (rows = GT x,y,z; cols = FE x,y,z):\n' + np.array2string(np.round(R, 2)))

    if only: fe_s = {k: v for k, v in fe_s.items() if k in only}; gt_s = {k: v for k, v in gt_s.items() if k in only}   # e.g. --only=mandible (initial alignment above still used the skull)
    mv = {k: np.asarray(f(np.asarray(v.points, dtype=float))) for k, v in fe_s.items()}
    fx = {k: np.asarray(v.points, dtype=float) for k, v in gt_s.items()}
    active = {k: np.ones(len(mv[k]), bool) for k in mv}
    # chin-based neck cut on the skin, same rule on both sides (each measured from its own mandible).
    # GT skin below the plane is removed; FE skin keeps its topology, points below the same plane (after the
    # first rigid CPD, inside fit) are inactive. The FE mandible is not used: its post-init chin differs by ~33 mm.
    cut_gt = None
    if 'skin' in fx:
        cut_gt = (fx['skin'][:, 2].max() - VERTEX_TO_CUT_MM) if skin_only else gt_ref['mandible_pts'][:, 2].min() - CUT_BELOW_CHIN_MM
        n0 = len(fx['skin']); fx['skin'] = fx['skin'][fx['skin'][:, 2] >= cut_gt]
        log.append(f'neck cut: GT z={cut_gt:.1f} removed {n0 - len(fx["skin"])} of {n0} skin points')
    # brain: the GT brain has a flat cut through the brainstem, FE brain continues below it
    if 'brain' in mv: active['brain'] = mv['brain'][:, 2] >= fx['brain'][:, 2].min() + 3

    summary, results = {}, {}
    for k in mv:
        ctx = fit(mv[k], fx[k], active[k], k, log, cut_z=cut_gt if k == 'skin' else None)
        final, conf, rows = refine(ctx); rows = {**ctx['rows'], **rows}
        for name, pos in (('5 TPS dense', ctx['warped']), ('6.5 blend (final)', final)):
            rows[name]['flipped_triangles'], ntri = flipped_triangles(fe_s[k], ctx['rigid'], pos)
        log.append(f'[{k}] flipped triangles: TPS {rows["5 TPS dense"]["flipped_triangles"]}, '
                   f'final {rows["6.5 blend (final)"]["flipped_triangles"]} of {ntri}')
        res = fe_s[k].copy(); res.points = final
        if skin_only and k == 'skin':   # chin completion: pull the nodes below the neck cut onto the input skin; gate metrics above use the pre-completion fit
            from chin_complete import complete_chin
            final = complete_chin(final, ctx['active'], np.asarray(gt_s['skin'].points, float)); res.points = final
            near = cKDTree(np.asarray(gt_s['skin'].points, float)).query(final)[0] < 8.0
            res.point_data['active_completed'] = (ctx['active'] | near).astype(np.uint8)
            # nose: the template nose is ~35 mm low and ~7 mm too flat after the fit; move its tip node onto the input nose tip (nose_landmark.py)
            from nose_landmark import find_nose, nose_fix
            g_all = np.asarray(gt_s['skin'].points, float); tip = g_all[find_nose(g_all)]
            nose_node = find_nose(f(np.asarray(fe_s['skin'].points, float)), (100.0, 175.0))
            final, d_tip = nose_fix(final, nose_node, tip, g_all); res.points = final
            log.append(f'nose fix: template nose node {nose_node} moved {np.linalg.norm(d_tip):.1f} mm onto the input nose tip')
            from mouth_flag import mouth_misfit
            mf = mouth_misfit(res, g_all); summary['mouth_misfit'] = mf
            log.append(f'mouth flag: {"MISFIT - do not trust lips/chin" if mf["flag"] else "ok"} (p95 {mf["p95_mm"]:.1f} mm, max {mf["max_mm"]:.1f} mm)')
        res.point_data['confidence'] = conf
        res.point_data['active'] = ctx['active'].astype(np.uint8)
        res.point_data['residual_mm'] = cKDTree(ctx['fx']).query(final)[0]
        res.point_data['rigid_pos_x'] = ctx['rigid'][:, 0]
        res.save(f'{out}/{case}_{k}_registered.vtp'); results[k] = res
        np.save(f'{out}/{case}_{k}_target_trimmed.npy', ctx['fx']); np.save(f'{out}/{case}_{k}_tps.npy', ctx['warped'])
        summary[k] = {st: {m: float(v) for m, v in r.items()} for st, r in rows.items()}
        summary[k]['n_triangles'] = ntri
        if k == 'skin':
            fin = rows['6.5 blend (final)']
            face_z = ctx['fx'][:, 2].max() - FACE_DEPTH_MM if skin_only else cut_gt
            fa = ctx['active'] & (final[:, 2] >= face_z); fg = ctx['fx'][ctx['fx'][:, 2] >= face_z]
            fin_face = agreement(final[fa], fg); fin_face['flipped_triangles'] = fin['flipped_triangles']
            summary['skin_face_region'] = {m: float(v) for m, v in fin_face.items()}
            d_gt = cKDTree(final[ctx['active']]).query(ctx['fx'])[0]    # GT -> FE: where the template has no skin
            gap = d_gt > COVER_GAP_MM; band = ctx['fx'][:, 2] < cut_gt + NECK_BAND_MM
            checks = dict(chamfer=fin_face['chamfer'] < GATE['chamfer'], hd95=fin_face['hd95'] < GATE['hd95'],
                          flips=fin['flipped_triangles'] / ntri < GATE['flip_frac'])
            summary['gate'] = dict(passed=bool(all(checks.values())), checks={c: bool(v) for c, v in checks.items()},
                                   coverage_gap_pct=float(100 * gap.mean()),
                                   coverage_gap_neck_band_pct=float(100 * (gap & band).sum() / len(gap)),
                                   coverage_gap_elsewhere_pct=float(100 * (gap & ~band).sum() / len(gap)),
                                   hausdorff_logged=float(fin['hausdorff']), face_chamfer=float(fin_face['chamfer']), face_hd95=float(fin_face['hd95']), whole_chamfer=float(fin['chamfer']), whole_hd95=float(fin['hd95']))
            g = summary['gate']
            log.append(f'GATE skin: {"PASS" if g["passed"] else "FAIL"} {g["checks"]} | GT skin >{COVER_GAP_MM:.0f} mm from FE: '
                       f'{g["coverage_gap_pct"]:.1f}% (neck band {g["coverage_gap_neck_band_pct"]:.1f}%, '
                       f'elsewhere {g["coverage_gap_elsewhere_pct"]:.1f}%); Hausdorff {g["hausdorff_logged"]:.1f} not gated')
    tag = ('_' + '_'.join(only)) if only else ''
    json.dump(summary, open(f'{out}/{case}{tag}_summary.json', 'w'), indent=1)
    open(f'{out}/{case}{tag}_log.txt', 'w').write('\n'.join(log))
    print('\n'.join(log))
    for k, rows in summary.items():
        if k in ('gate', 'skin_face_region'): continue
        print(f'\n== {k} ==')
        print(f'{"stage":20s} {"mean":>7s} {"median":>7s} {"hd95":>7s} {"chamfer":>8s} {"hausd":>7s}')
        for st, r in rows.items():
            if isinstance(r, dict):
                print(f'{st:20s} {r["mean"]:7.2f} {r["median"]:7.2f} {r["hd95"]:7.2f} {r["chamfer"]:8.2f} {r["hausdorff"]:7.2f}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    only = next((a.split('=')[1].split(',') for a in sys.argv if a.startswith('--only=')), None)
    main(args[0] if args else 'case_02', skin_only='--skin-only' in sys.argv, only=only)
