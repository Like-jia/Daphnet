# Daphnet

## Oxford Inertial Odometry Dataset (OxIOD) Integration

This repository includes tools for downloading and processing the [OxIOD dataset](https://arxiv.org/abs/1809.07491) — a benchmark dataset for deep inertial odometry, published by the University of Oxford.

### Quick Start

```bash
# 1. Download the dataset (requires internet access to deepio.cs.ox.ac.uk)
python scripts/download_oxiod.py

# 2. Inspect the dataset
python scripts/load_oxiod.py data/oxiod --category handheld
```

### Dataset Details

| Feature | Details |
|---------|---------|
| **Sensor** | BNO055 9-axis IMU (accel + gyro + mag) |
| **Rate** | 100 Hz (some at 200 Hz) |
| **Sequences** | 158 sequences, >42 km |
| **Ground Truth** | Vicon motion capture + Google Tango VIO |
| **Size** | ~1.01 GB compressed |

### Files

| File | Description |
|------|-------------|
| `scripts/download_oxiod.py` | Download script with fallback URLs and progress |
| `scripts/load_oxiod.py` | Data loader with sync, windowing, and feature extraction |
| `data/oxiod/README.md` | Detailed dataset documentation |

### Manual Download

If the automatic download doesn't work (e.g., the Oxford server is temporarily down):

1. Visit http://deepio.cs.ox.ac.uk in your browser
2. Download the dataset archive (~1.01 GB)
3. Extract to `data/oxiod/`
4. Run `python scripts/load_oxiod.py data/oxiod` to verify

See [data/oxiod/README.md](data/oxiod/README.md) for full documentation.