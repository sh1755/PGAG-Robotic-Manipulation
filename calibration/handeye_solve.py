"""Solve hand-eye calibration and report the residual you will cite in the paper.

Input: samples.json from collect_samples.py (or any source) with
    {"setup": "eye_in_hand" | "eye_to_hand",
     "samples": [{"T_base_ee": 4x4, "T_cam_target": 4x4}, ...]}

eye_in_hand : camera on the gripper, board fixed on the table.
              Solves X = T_ee_cam. Consistency: T_base_ee @ X @ T_cam_target is constant.
eye_to_hand : camera fixed, board held by / mounted on the gripper.
              Solves X = T_base_cam. Consistency: inv(T_base_ee) @ X @ T_cam_target is constant.

A pure-NumPy Park-Martin solver always runs; if your OpenCV has
cv2.calibrateHandEye, its five methods also run. The lowest residual wins.

    python -m calibration.handeye_solve calib/samples.json --out calib/handeye.json
    python -m calibration.handeye_solve --selftest          # synthetic check, no hardware
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
HAVE_CV_HANDEYE = hasattr(cv2, "calibrateHandEye")  # missing in some OpenCV 5 builds


def so3_log(R):
    """Rotation matrix -> axis-angle vector."""
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2 * np.sin(th)) * v


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def park_martin(G, C):
    """Solve G_i X C_i = const (AX = XB) with Park & Martin (1994). Pure NumPy.

    G: gripper->base transforms (or their inverses for eye-to-hand), C: target->camera.
    """
    A, B = [], []
    n = len(G)
    for i in range(n):
        for j in range(i + 1, n):
            A.append(inv(G[j]) @ G[i])
            B.append(C[j] @ inv(C[i]))
    M = sum(np.outer(so3_log(b[:3, :3]), so3_log(a[:3, :3])) for a, b in zip(A, B))
    U, S, Vt = np.linalg.svd(M.T @ M)
    Rx = U @ np.diag(1 / np.sqrt(S)) @ Vt @ M.T
    U2, _, Vt2 = np.linalg.svd(Rx)  # re-orthonormalise
    Rx = U2 @ Vt2
    if np.linalg.det(Rx) < 0:
        Rx = U2 @ np.diag([1, 1, -1]) @ Vt2
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    rhs = np.concatenate([Rx @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    tx = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return to_T(Rx, tx)


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3], Ti[:3, 3] = R.T, -R.T @ t
    return Ti


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).ravel()
    return T


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def mat_to_quat_xyzw(R):
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q[3], q[0], q[1], q[2] = s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q[i] = s / 4
        q[3] = (R[k, j] - R[j, k]) / s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def solve(T_base_ee, T_cam_target, setup):
    if setup == "eye_in_hand":
        A = T_base_ee  # gripper -> base
    else:
        A = [inv(T) for T in T_base_ee]  # base -> gripper, gives X = T_base_cam
    Rg = [T[:3, :3] for T in A]
    tg = [T[:3, 3] for T in A]
    Rt = [T[:3, :3] for T in T_cam_target]
    tt = [T[:3, 3] for T in T_cam_target]
    results = {"PARK_NUMPY": None}
    X = park_martin(A, T_cam_target)
    results["PARK_NUMPY"] = (X, residual(X, T_base_ee, T_cam_target, setup))
    if not HAVE_CV_HANDEYE:
        return results
    for name, m in METHODS.items():
        try:
            R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m)
        except cv2.error:
            continue
        X = to_T(R, t)
        results[name] = (X, residual(X, T_base_ee, T_cam_target, setup))
    return results


def residual(X, T_base_ee, T_cam_target, setup):
    chain = []
    for Tb, Tc in zip(T_base_ee, T_cam_target):
        chain.append(Tb @ X @ Tc if setup == "eye_in_hand" else inv(Tb) @ X @ Tc)
    t = np.array([C[:3, 3] for C in chain])
    t_err = np.linalg.norm(t - t.mean(0), axis=1) * 1e3
    R0 = chain[0][:3, :3]
    r_err = np.array([rot_angle_deg(R0.T @ C[:3, :3]) for C in chain])
    return {"trans_rms_mm": float(np.sqrt(np.mean(t_err ** 2))), "trans_max_mm": float(t_err.max()),
            "rot_mean_deg": float(r_err.mean()), "rot_max_deg": float(r_err.max())}


def random_T(rng, t_scale=0.3, rot_deg=40):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = so3_exp(axis * np.radians(rng.uniform(-rot_deg, rot_deg)))
    return to_T(R, rng.uniform(-t_scale, t_scale, 3))


def selftest():
    rng = np.random.default_rng(0)
    X_true = to_T(so3_exp(np.array([0.1, -0.2, 0.05])), [0.05, -0.03, 0.08])  # T_ee_cam
    T_base_target = to_T(np.eye(3), [0.6, 0.0, 0.0])
    Tb, Tc = [], []
    for _ in range(20):
        T_base_ee = random_T(rng) @ to_T(np.eye(3), [0.4, 0, 0.4])
        T_cam_target = inv(X_true) @ inv(T_base_ee) @ T_base_target
        noise = to_T(so3_exp(rng.normal(0, np.radians(0.1), 3)), rng.normal(0, 0.0005, 3))
        Tb.append(T_base_ee)
        Tc.append(T_cam_target @ noise)
    res = solve(Tb, Tc, "eye_in_hand")
    print("Synthetic eye-in-hand, 20 poses, 0.5 mm / 0.1 deg noise")
    for name, (X, r) in res.items():
        err_t = np.linalg.norm(X[:3, 3] - X_true[:3, 3]) * 1e3
        err_r = rot_angle_deg(X[:3, :3].T @ X_true[:3, :3])
        print(f"  {name:<11} X error {err_t:5.2f} mm {err_r:5.2f} deg | residual {r['trans_rms_mm']:.2f} mm rms")
    best = min(res.items(), key=lambda kv: kv[1][1]["trans_rms_mm"])
    ok = np.linalg.norm(best[1][0][:3, 3] - X_true[:3, 3]) < 0.003
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="?")
    ap.add_argument("--out", default="calib/handeye.json")
    ap.add_argument("--parent-frame", default="fr3_hand", help="TF parent for the printed static transform")
    ap.add_argument("--child-frame", default="camera_color_optical_frame")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not args.samples:
        ap.error("give a samples.json or --selftest")

    data = json.load(open(args.samples))
    setup = data["setup"]
    Tb = [np.array(s["T_base_ee"]) for s in data["samples"]]
    Tc = [np.array(s["T_cam_target"]) for s in data["samples"]]
    if len(Tb) < 10:
        print(f"WARNING: only {len(Tb)} samples; aim for 15-25 with large rotation diversity.")

    res = solve(Tb, Tc, setup)
    print(f"{setup}, {len(Tb)} samples")
    print(f"{'method':<11}{'rms mm':>9}{'max mm':>9}{'mean deg':>10}{'max deg':>9}")
    for name, (_, r) in res.items():
        print(f"{name:<11}{r['trans_rms_mm']:>9.2f}{r['trans_max_mm']:>9.2f}{r['rot_mean_deg']:>10.2f}{r['rot_max_deg']:>9.2f}")
    name, (X, r) = min(res.items(), key=lambda kv: kv[1][1]["trans_rms_mm"])
    q = mat_to_quat_xyzw(X[:3, :3])
    out = {"setup": setup, "method": name, "n_samples": len(Tb), "T": X.tolist(),
           "translation_xyz": X[:3, 3].tolist(), "quaternion_xyzw": q.tolist(), "residual": r,
           "frames": {"parent": args.parent_frame if setup == "eye_in_hand" else "fr3_link0",
                      "child": args.child_frame}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    x, y, z = X[:3, 3]
    print(f"\nBest: {name}. Saved {args.out}")
    print("Publish it in ROS2 with:")
    print(f"  ros2 run tf2_ros static_transform_publisher --x {x:.5f} --y {y:.5f} --z {z:.5f} "
          f"--qx {q[0]:.6f} --qy {q[1]:.6f} --qz {q[2]:.6f} --qw {q[3]:.6f} "
          f"--frame-id {out['frames']['parent']} --child-frame-id {args.child_frame}")
    if r["trans_rms_mm"] > 3.0:
        print("\nResidual > 3 mm: check board size settings, add samples with larger rotations, re-check intrinsics.")


if __name__ == "__main__":
    main()
