"""Stack the per-case FE-topology skin / skull / brain / mandible meshes into one array for the shape model.

Skin comes from the skin-only run (what exists at deployment); skull and brain come from the run that saw their
own GT (training targets). Cases are aligned with a rigid transform (no scale) driven by reliable SKIN nodes only,
then the same transform is applied to skull and brain, so the model never needs the internal structures to pose a case.
"""
import glob, json, os
import numpy as np, pyvista as pv

BASE = '/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/outputs'
SKIN_DIR, FULL_DIR = f'{BASE}/hanseg_template_reg_skinonly', f'{BASE}/hanseg_template_reg'
OUT = f'{BASE}/shape_dataset'
EXCLUDE = {'case_09'}             # user decision: not used
RELIABLE_FRAC, BRAIN_RES_MM = 0.9, 8.0


def kabsch(src, dst):
    """Rigid (R, t) minimising |src @ R.T + t - dst|, no scale."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - cs @ R.T


def load(case):
    rd = lambda d, k: pv.read(f'{d}/{case}/{case}_{k}_registered.vtp')
    s, k, b, md = rd(SKIN_DIR, 'skin'), rd(FULL_DIR, 'skull'), rd(FULL_DIR, 'brain'), rd(FULL_DIR, 'mandible')
    g = lambda m, a: np.asarray(m.point_data[a])
    return dict(skin=np.asarray(s.points), skull=np.asarray(k.points), brain=np.asarray(b.points), mandible=np.asarray(md.points),
                mandible_active=g(md, 'active').astype(bool), mandible_res=g(md, 'residual_mm'),
                skin_active=g(s, 'active_completed' if 'active_completed' in s.array_names else 'active').astype(bool), brain_active=g(b, 'active').astype(bool),
                brain_res=g(b, 'residual_mm'), skull_res=g(k, 'residual_mm'))


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = [os.path.basename(d) for d in sorted(glob.glob(f'{SKIN_DIR}/case_*'))]
    cases = [c for c in cases if c not in EXCLUDE and os.path.exists(f'{FULL_DIR}/{c}/{c}_brain_registered.vtp')
             and os.path.exists(f'{FULL_DIR}/{c}/{c}_mandible_registered.vtp')]
    D = [load(c) for c in cases]
    n = len(cases)

    skin_rel = np.mean([d['skin_active'] for d in D], axis=0) >= RELIABLE_FRAC
    brain_rel = (np.mean([d['brain_active'] for d in D], axis=0) >= RELIABLE_FRAC) & \
                (np.median([d['brain_res'] for d in D], axis=0) < BRAIN_RES_MM)
    skull_rel = np.ones(len(D[0]['skull']), bool)
    mand_rel = (np.mean([d['mandible_active'] for d in D], axis=0) >= RELIABLE_FRAC) & (np.median([d['mandible_res'] for d in D], axis=0) < BRAIN_RES_MM)

    # rigid alignment on reliable skin nodes only; reference = running mean, 5 rounds
    ref = D[0]['skin'][skin_rel]
    for _ in range(5):
        T = [kabsch(d['skin'][skin_rel], ref) for d in D]
        ref = np.mean([d['skin'][skin_rel] @ R.T + t for d, (R, t) in zip(D, T)], axis=0)
    out = {k: np.stack([d[k] @ R.T + t for d, (R, t) in zip(D, T)]) for k in ('skin', 'skull', 'brain', 'mandible')}
    resid = np.linalg.norm(out['skin'][:, skin_rel] - ref, axis=2)
    np.savez_compressed(f'{OUT}/shape_dataset.npz', cases=np.array(cases), skin_reliable=skin_rel,
                        skull_reliable=skull_rel, brain_reliable=brain_rel, mandible_reliable=mand_rel,
                        R=np.stack([R for R, _ in T]), t=np.stack([t for _, t in T]), **out)   # aligned = raw @ R.T + t
    info = dict(n_cases=n, excluded=sorted(EXCLUDE), nodes={k: int(out[k].shape[1]) for k in out},
                reliable={'skin': int(skin_rel.sum()), 'skull': int(skull_rel.sum()), 'brain': int(brain_rel.sum()), 'mandible': int(mand_rel.sum())},
                skin_rms_to_mean_mm=float(np.sqrt((resid ** 2).mean())))
    json.dump(info, open(f'{OUT}/shape_dataset_info.json', 'w'), indent=1)
    print(json.dumps(info, indent=1))

    # quick structure check: variance explained and size correlation (not a validation)
    for k, m in (('skin', skin_rel), ('skull', skull_rel), ('brain', brain_rel), ('mandible', mand_rel)):
        X = out[k][:, m].reshape(n, -1); X = X - X.mean(0)
        s = np.linalg.svd(X, compute_uv=False) ** 2; c = np.cumsum(s) / s.sum()
        print(f'{k:5s} PCA modes for 80/90/95% variance: {int((c < .8).sum()) + 1}/{int((c < .9).sum()) + 1}/{int((c < .95).sum()) + 1}')


if __name__ == '__main__':
    main()
