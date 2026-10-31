"""Collect hand-eye samples: RealSense + ChArUco board + robot pose from ROS2 TF.

Run with the robot driver publishing TF (e.g. franka_ros2 bringup), then:

    python calibration/collect_samples.py --setup eye_in_hand \
        --base-frame fr3_link0 --ee-frame fr3_hand --out calib/samples.json

Move the robot (guiding mode is easiest) to a new pose, keep it still, press
ENTER to capture. Collect 15-25 poses with large rotation differences
(at least +/-30 deg about each axis) and the whole board visible. Type q + ENTER
to save and quit.

Board defaults match a printed 5x7 ChArUco (square 40 mm, marker 30 mm,
DICT_5X5_100). Generate one with --make-board board.png and print at 100%,
then MEASURE the printed square size and pass it with --square.

Needs: pyrealsense2, opencv-contrib-python (>= 4.7 preferred), rclpy, tf2_ros.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def make_board(args):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    if hasattr(cv2.aruco, "CharucoBoard") and not hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard((args.cols, args.rows), args.square, args.marker, d)
    else:  # OpenCV < 4.7
        board = cv2.aruco.CharucoBoard_create(args.cols, args.rows, args.square, args.marker, d)
    return d, board


def detect_board_pose(img, d, board, K, dist):
    """Return 4x4 T_cam_target, or None. Works with old and new OpenCV aruco APIs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "CharucoDetector"):
        ch_corners, ch_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
        obj_all = board.getChessboardCorners()
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d)
        if ids is None:
            return None
        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        obj_all = board.chessboardCorners
    if ch_ids is None or len(ch_ids) < 8:
        return None
    obj = np.array([obj_all[i] for i in ch_ids.ravel()], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, ch_corners.reshape(-1, 2).astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", choices=["eye_in_hand", "eye_to_hand"], default="eye_in_hand")
    ap.add_argument("--base-frame", default="fr3_link0")
    ap.add_argument("--ee-frame", default="fr3_hand")
    ap.add_argument("--out", default="calib/samples.json")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--square", type=float, default=0.040, help="measured square size (m)")
    ap.add_argument("--marker", type=float, default=0.030, help="measured marker size (m)")
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--make-board", metavar="PNG", help="write a printable board image and exit")
    args = ap.parse_args()

    d, board = make_board(args)
    if args.make_board:
        px = 100  # pixels per square
        size = (args.cols * px, args.rows * px)
        img = board.generateImage(size) if hasattr(board, "generateImage") else board.draw(size)
        cv2.imwrite(args.make_board, img)
        print(f"Wrote {args.make_board}. Print at 100% scale, then measure square and marker sizes.")
        return

    import pyrealsense2 as rs
    import rclpy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs[:5])
    print(f"RealSense intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f} model={intr.model}")

    rclpy.init()
    node = rclpy.create_node("handeye_collector")
    buf = Buffer()
    TransformListener(buf, node)

    samples = []
    print("Move robot, hold still, press ENTER to capture. 'q' + ENTER to save and quit.")
    try:
        while True:
            line = sys.stdin.readline().strip()
            if line == "q":
                break
            for _ in range(5):  # let auto-exposure settle, use the latest frame
                frames = pipe.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            T_cam_target = detect_board_pose(img, d, board, K, dist)
            if T_cam_target is None:
                print("  board not detected (need >= 8 corners). Try again.")
                continue
            end = time.time() + 1.0
            tf = None
            while time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
                if buf.can_transform(args.base_frame, args.ee_frame, Time()):
                    tf = buf.lookup_transform(args.base_frame, args.ee_frame, Time())
                    break
            if tf is None:
                print(f"  no TF {args.base_frame} -> {args.ee_frame}. Is the robot driver running?")
                continue
            tr, q = tf.transform.translation, tf.transform.rotation
            T_base_ee = np.eye(4)
            T_base_ee[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
            T_base_ee[:3, 3] = [tr.x, tr.y, tr.z]
            samples.append({"T_base_ee": T_base_ee.tolist(), "T_cam_target": T_cam_target.tolist()})
            print(f"  captured sample {len(samples)} (board at {np.linalg.norm(T_cam_target[:3, 3]):.2f} m)")
            cv2.imwrite(str(Path(args.out).with_suffix("")) + f"_{len(samples):02d}.png", img)
    finally:
        pipe.stop()
        node.destroy_node()
        rclpy.shutdown()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"setup": args.setup, "base_frame": args.base_frame, "ee_frame": args.ee_frame,
                   "intrinsics": {"K": K.tolist(), "dist": dist.tolist()}, "samples": samples},
                  open(args.out, "w"), indent=2)
        print(f"Saved {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
