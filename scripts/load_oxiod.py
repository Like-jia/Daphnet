#!/usr/bin/env python3
"""
Data loader for the Oxford Inertial Odometry Dataset (OxIOD).

Reads OxIOD CSV files and provides functions to:
- Load IMU data (accelerometer, gyroscope, magnetometer, orientation)
- Load ground truth trajectories (Vicon / Google Tango)
- Synchronize IMU and ground truth timestamps
- Extract windowed segments for CNN/PINN training

Reference: https://arxiv.org/abs/1809.07491
Based on dataset structure from:
  - https://github.com/BehnamZeinali/IMUNet
  - https://github.com/jpsml/6-DOF-Inertial-Odometry

OxIOD IMU CSV columns (100 Hz):
  Col 0:  timestamp (ns)
  Col 1:  header.seq
  Col 2:  header.stamp.secs
  Col 3:  header.stamp.nsecs
  Col 4:  angular_velocity.x (gyro_x, rad/s)
  Col 5:  angular_velocity.y (gyro_y, rad/s)
  Col 6:  angular_velocity.z (gyro_z, rad/s)
  Col 7:  magnetic_field.x
  Col 8:  magnetic_field.y
  Col 9:  magnetic_field.z
  Col 10: linear_acceleration.x (acc_x, m/s^2)
  Col 11: linear_acceleration.y (acc_y, m/s^2)
  Col 12: linear_acceleration.z (acc_z, m/s^2)
  Col 13: orientation.x (quat_x)
  Col 14: orientation.y (quat_y)
  Col 15: orientation.z (quat_z)
  Col 16: orientation.w (quat_w)
  Col 17-19: gravity vector (grav_x, grav_y, grav_z)
  Col 20: roll
  Col 21: pitch
  Col 22: yaw

OxIOD Ground Truth CSV columns (Vicon / VIO):
  Col 0: timestamp (ns)
  Col 1: header.seq
  Col 2: position.x
  Col 3: position.y
  Col 4: position.z
  Col 5: orientation.x
  Col 6: orientation.y
  Col 7: orientation.z
  Col 8: orientation.w
"""

import os
import glob
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import scipy.interpolate
except ImportError:
    scipy = None


# ---- Dataset categories and their standard train/test sequences ----
DATASET_CATEGORIES = [
    "handbag",
    "handheld",
    "large scale",
    "multi devices",
    "multi users",
    "pocket",
    "running",
    "slow walking",
    "trolley",
]

# Activity label mapping (for 6-class classification)
ACTIVITY_LABELS = {
    "slow walking": 0,
    "handheld": 1,       # normal walking (handheld)
    "pocket": 2,         # normal walking (pocket)
    "handbag": 3,        # normal walking (handbag)
    "running": 4,
    "large scale": 5,    # mixed / long-distance
}


def load_imu_csv(filepath):
    """
    Load an OxIOD IMU CSV file.

    Parameters
    ----------
    filepath : str
        Path to the imu*.csv file.

    Returns
    -------
    dict with keys:
        'timestamp'   : (N,) array, timestamps in seconds
        'gyro'        : (N, 3) array, angular velocity (rad/s)
        'acc'         : (N, 3) array, linear acceleration (m/s^2)
        'mag'         : (N, 3) array, magnetic field
        'orientation' : (N, 4) array, quaternion (x, y, z, w)
        'gravity'     : (N, 3) array, gravity vector (if available)
        'euler'       : (N, 3) array, roll/pitch/yaw (if available)
    """
    if pd is None:
        raise ImportError("pandas is required: pip install pandas")

    data = pd.read_csv(filepath).values

    result = {
        "timestamp": data[:, 0] / 1e9,  # Convert ns to seconds
        "gyro": data[:, 4:7].astype(np.float64),
        "acc": data[:, 10:13].astype(np.float64),
        "mag": data[:, 7:10].astype(np.float64),
        "orientation": data[:, 13:17].astype(np.float64),  # qx, qy, qz, qw
    }

    # Gravity and Euler angles may not be present in all files
    if data.shape[1] > 19:
        result["gravity"] = data[:, 17:20].astype(np.float64)
    if data.shape[1] > 22:
        result["euler"] = data[:, 20:23].astype(np.float64)  # roll, pitch, yaw

    return result


def load_ground_truth_csv(filepath):
    """
    Load an OxIOD ground truth CSV file (Vicon or VIO).

    Parameters
    ----------
    filepath : str
        Path to the vi*.csv or hand*.csv ground truth file.

    Returns
    -------
    dict with keys:
        'timestamp'   : (N,) array, timestamps in seconds
        'position'    : (N, 3) array, position (x, y, z) in meters
        'orientation' : (N, 4) array, quaternion (x, y, z, w)
    """
    if pd is None:
        raise ImportError("pandas is required: pip install pandas")

    data = pd.read_csv(filepath).values

    return {
        "timestamp": data[:, 0] / 1e9,  # Convert ns to seconds
        "position": data[:, 2:5].astype(np.float64),
        "orientation": data[:, 5:9].astype(np.float64),  # qx, qy, qz, qw
    }


def synchronize_imu_gt(imu_data, gt_data):
    """
    Synchronize IMU and ground truth data by interpolating IMU
    measurements to ground truth timestamps.

    Parameters
    ----------
    imu_data : dict
        Output from load_imu_csv().
    gt_data : dict
        Output from load_ground_truth_csv().

    Returns
    -------
    dict with synchronized data:
        'timestamp' : (N,) array
        'gyro'      : (N, 3) array
        'acc'       : (N, 3) array
        'position'  : (N, 3) array
        'orientation': (N, 4) array
    """
    if scipy is None:
        raise ImportError("scipy is required: pip install scipy")

    imu_ts = imu_data["timestamp"]
    gt_ts = gt_data["timestamp"]

    # Find overlapping time range
    t_start = max(imu_ts[0], gt_ts[0])
    t_end = min(imu_ts[-1], gt_ts[-1])

    # Filter ground truth to overlapping range
    mask = (gt_ts >= t_start) & (gt_ts <= t_end)
    sync_ts = gt_ts[mask]

    if len(sync_ts) == 0:
        raise ValueError("No overlapping timestamps between IMU and ground truth")

    # Interpolate IMU data to ground truth timestamps
    sync_gyro = scipy.interpolate.interp1d(
        imu_ts, imu_data["gyro"], axis=0
    )(sync_ts)
    sync_acc = scipy.interpolate.interp1d(
        imu_ts, imu_data["acc"], axis=0
    )(sync_ts)

    return {
        "timestamp": sync_ts,
        "gyro": sync_gyro,
        "acc": sync_acc,
        "position": gt_data["position"][mask],
        "orientation": gt_data["orientation"][mask],
    }


def find_sequence_pairs(root_dir, category="handheld"):
    """
    Find all (imu_file, gt_file) pairs in a dataset category.

    Parameters
    ----------
    root_dir : str
        Root directory of the OxIOD dataset.
    category : str
        Dataset category (e.g., 'handheld', 'pocket', etc.).

    Returns
    -------
    list of (imu_path, gt_path) tuples
    """
    pairs = []
    cat_dir = os.path.join(root_dir, category)

    if not os.path.isdir(cat_dir):
        return pairs

    for data_dir in sorted(glob.glob(os.path.join(cat_dir, "data*"))):
        raw_dir = os.path.join(data_dir, "raw")
        syn_dir = os.path.join(data_dir, "syn")

        # Prefer synchronized data if available, otherwise use raw
        search_dir = syn_dir if os.path.isdir(syn_dir) else raw_dir
        if not os.path.isdir(search_dir):
            search_dir = data_dir

        for imu_file in sorted(glob.glob(os.path.join(search_dir, "imu*.csv"))):
            # Determine corresponding ground truth file
            basename = os.path.basename(imu_file)
            seq_num = basename.replace("imu", "").replace(".csv", "")

            # Try different GT naming conventions
            gt_candidates = [
                os.path.join(search_dir, f"vi{seq_num}.csv"),
                os.path.join(search_dir, f"hand{seq_num}.csv"),
            ]

            # For large scale / floor sequences, GT may be in a tango subfolder
            tango_dir = os.path.join(data_dir, "tango")
            if os.path.isdir(tango_dir):
                gt_candidates.append(
                    os.path.join(tango_dir, f"tango{seq_num}.csv")
                )

            for gt_file in gt_candidates:
                if os.path.isfile(gt_file):
                    pairs.append((imu_file, gt_file))
                    break

    return pairs


def extract_windows(data, window_size=128, stride=64):
    """
    Extract sliding windows from synchronized data.

    Parameters
    ----------
    data : dict
        Output from synchronize_imu_gt().
    window_size : int
        Window size in samples (default: 128 ≈ 1.28s at 100Hz).
    stride : int
        Step size between windows (default: 64, 50% overlap).

    Returns
    -------
    X : (num_windows, window_size, 6) array
        Windowed IMU data [gyro_xyz, acc_xyz].
    positions : (num_windows, 3) array
        Position at center of each window (for trajectory evaluation).
    """
    gyro = data["gyro"]
    acc = data["acc"]
    pos = data["position"]

    imu_combined = np.concatenate([gyro, acc], axis=1)  # (N, 6)

    n_samples = imu_combined.shape[0]
    windows = []
    center_positions = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(imu_combined[start:end])
        center_positions.append(pos[start + window_size // 2])

    if not windows:
        return np.array([]), np.array([])

    X = np.array(windows)
    positions = np.array(center_positions)

    return X, positions


def load_all_sequences(root_dir, categories=None, window_size=128, stride=64):
    """
    Load all sequences from specified categories and extract windows.

    Parameters
    ----------
    root_dir : str
        Root directory of the OxIOD dataset.
    categories : list of str, optional
        Categories to load. Defaults to all available categories.
    window_size : int
        Window size for feature extraction.
    stride : int
        Stride for sliding window.

    Returns
    -------
    X : array of shape (total_windows, window_size, 6)
    y : array of shape (total_windows,) activity labels
    positions : array of shape (total_windows, 3)
    """
    if categories is None:
        categories = DATASET_CATEGORIES

    all_X = []
    all_y = []
    all_pos = []

    for category in categories:
        label = ACTIVITY_LABELS.get(category)
        if label is None:
            continue

        pairs = find_sequence_pairs(root_dir, category)
        for imu_file, gt_file in pairs:
            try:
                imu_data = load_imu_csv(imu_file)
                gt_data = load_ground_truth_csv(gt_file)
                synced = synchronize_imu_gt(imu_data, gt_data)
                X, positions = extract_windows(synced, window_size, stride)

                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(np.full(len(X), label, dtype=np.int64))
                    all_pos.append(positions)
                    print(
                        f"  Loaded {len(X)} windows from {os.path.basename(imu_file)} "
                        f"[{category}]"
                    )
            except Exception as e:
                print(f"  Warning: Failed to load {imu_file}: {e}")

    if not all_X:
        raise RuntimeError(
            f"No data loaded from {root_dir}. "
            "Please ensure the OxIOD dataset is downloaded and extracted correctly."
        )

    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_pos, axis=0),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and inspect OxIOD dataset"
    )
    parser.add_argument(
        "root_dir",
        help="Root directory of the OxIOD dataset",
    )
    parser.add_argument(
        "--category",
        default="handheld",
        help="Category to inspect (default: handheld)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Window size in samples (default: 128)",
    )
    args = parser.parse_args()

    print(f"Scanning OxIOD dataset at: {args.root_dir}")
    print(f"Category: {args.category}")
    print()

    pairs = find_sequence_pairs(args.root_dir, args.category)
    print(f"Found {len(pairs)} sequence pairs:")
    for imu_f, gt_f in pairs:
        print(f"  IMU: {os.path.relpath(imu_f, args.root_dir)}")
        print(f"  GT:  {os.path.relpath(gt_f, args.root_dir)}")
        print()

    if pairs:
        print("Loading first sequence...")
        imu_data = load_imu_csv(pairs[0][0])
        gt_data = load_ground_truth_csv(pairs[0][1])
        print(f"  IMU samples: {len(imu_data['timestamp'])}")
        print(f"  GT samples:  {len(gt_data['timestamp'])}")
        print(f"  IMU duration: {imu_data['timestamp'][-1] - imu_data['timestamp'][0]:.1f}s")
        print(f"  Gyro range: [{imu_data['gyro'].min():.3f}, {imu_data['gyro'].max():.3f}] rad/s")
        print(f"  Acc range: [{imu_data['acc'].min():.3f}, {imu_data['acc'].max():.3f}] m/s²")

        synced = synchronize_imu_gt(imu_data, gt_data)
        X, positions = extract_windows(synced, args.window_size)
        print(f"  Windows extracted: {len(X)} (window_size={args.window_size})")
        print(f"  Window shape: {X.shape}")
