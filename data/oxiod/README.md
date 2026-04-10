# Oxford Inertial Odometry Dataset (OxIOD)

## Overview

The OxIOD dataset is a benchmark dataset for deep inertial odometry research,
published by the University of Oxford's Department of Computer Science.

- **Paper**: [OxIOD: The Dataset for Deep Inertial Odometry](https://arxiv.org/abs/1809.07491)
- **Official Site**: http://deepio.cs.ox.ac.uk
- **Size**: ~1.01 GB (compressed)

## Key Features

| Feature | Details |
|---------|---------|
| **Sensor** | BNO055 9-axis IMU (accelerometer + gyroscope + magnetometer) |
| **Sampling Rate** | 100 Hz (some sequences at 200 Hz) |
| **Total Sequences** | 158 sequences, >42 km of inertial data |
| **Ground Truth** | Vicon motion capture (indoor) + Google Tango VIO |
| **Data Format** | CSV files with timestamps |

## Dataset Structure

```
oxiod/
├── Handbag/           # IMU in handbag scenarios
│   ├── data1/
│   │   ├── imu*.csv
│   │   └── vi*.csv    # Ground truth (Vicon/VIO)
│   └── ...
├── Handheld/          # IMU handheld scenarios
├── Pocket/            # IMU in pocket scenarios
├── Trolley/           # IMU on trolley scenarios
├── Slow Walking/      # Slow walking scenarios
├── Running/           # Running scenarios
└── ...
```

### Data Columns (IMU CSV)

Each `imu*.csv` file contains:
- `timestamp` — Unix timestamp (seconds)
- `ax, ay, az` — 3-axis accelerometer (m/s²)
- `gx, gy, gz` — 3-axis gyroscope (rad/s)
- `mx, my, mz` — 3-axis magnetometer
- `roll, pitch, yaw` — Orientation angles
- `grav_x, grav_y, grav_z` — Gravity vector components

### Ground Truth CSV

Each `vi*.csv` file contains:
- `timestamp` — Unix timestamp
- `px, py, pz` — 3D position
- `qw, qx, qy, qz` — Quaternion orientation

## Compatibility with ZUPT/ESKF Pipeline

This dataset is well-suited for foot-mounted inertial navigation research:

- **100 Hz sampling rate** → compatible with `window_size=128` (≈1.28s window)
- **Full 9-axis IMU data** → directly usable with `extract_physics_features()`
- **Multiple activity types** → walking, running, stairs → maps to 6-class classification
- **High-precision ground truth** → Vicon (sub-mm indoor) for ESKF validation

## Download

### Automatic Download

```bash
python scripts/download_oxiod.py
```

### Manual Download

1. Visit http://deepio.cs.ox.ac.uk
2. Download the dataset archive (~1.01 GB)
3. Extract contents to this directory (`data/oxiod/`)

## Citation

```bibtex
@inproceedings{chen2018oxiod,
  title={OxIOD: The Dataset for Deep Inertial Odometry},
  author={Chen, Changhao and Zhao, Peijun and Lu, Chris Xiaoxuan and
          Wang, Wei and Markham, Andrew and Trigoni, Niki},
  booktitle={arXiv preprint arXiv:1809.07491},
  year={2018}
}
```
