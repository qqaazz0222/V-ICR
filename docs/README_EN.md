# V-ICR: Video Iterative Context Refinement

**Video-based Action Recognition Pipeline** - Action recognition system using YOLO + ByteTrack tracking and Qwen3-VL based iterative context refinement

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[🇰🇷 한국어 README](../README.md)

## 📋 Overview

V-ICR is an end-to-end pipeline that detects and tracks people in video, then recognizes each person's actions using Vision-Language Model (VLM).

### Key Features

- **🎯 Precise Person Tracking**: Robust multi-object tracking based on YOLO12 + ByteTrack
- **🔧 Track Post-processing**: Kalman filter smoothing, broken track stitching
- **🧠 VLM-based Action Recognition**: Per-second action classification using Qwen3-VL-8B
- **🔄 Temporal Soft-label Refinement**: Iterative refinement using soft-label candidates and temporal context
- **📊 Similar Action Grouping**: VLM-based automatic action categorization and labelmap generation
- **📦 Unified Label Output**: Frame-level bbox + action label integrated data

## 🏗️ System Architecture

```
Input Video (MP4)
        │
        ▼
┌───────────────────┐
│    Detector       │ ← YOLO12 + ByteTrack
│  (Detection &     │
│   Tracking)       │
└─────────┬─────────┘
          │
          ▼
    ┌───────────┐
    │   Tubes   │ ← Per-person cropped videos
    └─────┬─────┘
          │
          ▼
┌───────────────────┐
│   Recognizer      │ ← Qwen3-VL-8B
│  (Action          │
│   Recognition)    │
│                   │
│  1. Per-second    │
│     analysis      │
│  2. Soft-label    │
│     top 5         │
│  3. Similar action│
│     grouping      │
│  4. Labelmap      │
│     refinement    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    Exporter       │
│  (Label Data      │
│   Generation)     │
└─────────┬─────────┘
          │
          ▼
  output/<video>.json
```

## 📁 Project Structure

```
V-ICR/
├── run.py                    # Main execution script
├── requirements.txt          # Python dependencies
├── checkpoints/              # Model weights
│   └── yolo12x.pt           # YOLO12 weights
├── modules/                  # Core modules
│   ├── detector.py          # Detection and tracking module
│   ├── recognizer.py        # Action recognition module
│   ├── exporter.py          # Label data export module
│   ├── dataset.py           # Dataset utilities
│   └── bytetrack_tuned.yaml # ByteTrack configuration
├── utils/                    # Utilities
│   └── logger.py            # Logging utility
├── data/                     # Data directory
│   ├── input/               # Input videos (place MP4 files here)
│   ├── working/             # Intermediate results
│   └── output/              # Final label data output
└── docs/                     # Documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Model Preparation

Place YOLO12 weights in the `checkpoints/` directory.
Qwen3-VL-8B will be automatically downloaded from HuggingFace on first run.

### 3. Process Video

```bash
# Place video in input directory
cp your_video.mp4 data/input/

# Run processing
python run.py
```

### 4. Predefined Label Map (Optional)

You can specify predefined action labels in `data/input/label_map.txt`.
If this file exists and contains labels, only those labels will be used during recognition.

```
# label_map.txt example (both formats supported)

# Format 1: Labels only
standing
walking
running
punching
blocking

# Format 2: Number: Label
0: standing
1: walking
2: running
```

### 5. Check Results

**Intermediate Results** (`data/working/<video_name>/`):
- `tubes/` - Per-person cropped videos
- `tubes/metadata.json` - Tracking metadata
- `tubes/recognition_results.json` - Action recognition results
- `label_map.txt` - Action category mapping

**Final Output** (`data/output/`):
- `<video_name>.json` - Frame-level integrated label data

## ⚙️ Command Line Options

```bash
python run.py [OPTIONS]

Options:
  --skip-recognition          Skip action recognition phase (detection only)
  --refinement-iterations N   Number of refinement iterations (default: 5)
  --input-dir DIR             Input video directory (default: ./data/input)
  --working-dir DIR           Working directory (default: ./data/working)
  --output-dir DIR            Output directory (default: ./data/output)
```

**Examples:**

```bash
# Detection only
python run.py --skip-recognition

# 3 refinement iterations
python run.py --refinement-iterations 3

# Custom directories
python run.py --input-dir ./my_videos --output-dir ./my_output
```

## 📊 Output Format

### Final Label Data (`output/<video>.json`)

Integrated data containing frame-level bbox and action labels:

```json
{
  "version": "1.0",
  "video": {
    "name": "demo",
    "fps": 120.08,
    "width": 1920,
    "height": 1080,
    "total_frames": 1521,
    "duration": 12.67
  },
  "labelmap": {
    "0": {"name": "attacking", "original_actions": ["punching", ...]},
    "1": {"name": "boxing practice", "original_actions": [...]}
  },
  "num_persons": 3,
  "persons": {
    "id_1": {
      "action_summary": ["defending", "boxing practice"],
      "action_timeline": {
        "0": {"action": "standing", "action_id": 15}
      },
      "frames": [
        {
          "frame_idx": 0,
          "timestamp": 0.0,
          "bbox": {"x1": 1006, "y1": 157, "x2": 1275, "y2": 599},
          "action": "standing and moving",
          "action_id": 15
        }
      ]
    }
  },
  "summary": {
    "total_action_instances": 36,
    "action_distribution": {"boxing practice": 16, "defending": 3}
  }
}
```

### label_map.txt

Similar action grouping results:

```
0: attacking
   -> punching, throwing punch, raising arms
1: boxing practice
   -> boxing, sparring, shadow boxing
2: defending
   -> blocking, defending, preparing to block
```

### recognition_results.json (Intermediate)

```json
{
  "video_path": "./data/input/demo.mp4",
  "labelmap": ["attacking", "boxing practice", "defending", ...],
  "action_groups": {
    "boxing practice": ["boxing", "sparring", "shadow boxing"]
  },
  "tubes": {
    "id_1": {
      "temporal_labels": {
        "0": [
          {"action": "standing", "confidence": 0.90},
          {"action": "looking", "confidence": 0.75}
        ]
      },
      "final_actions": [
        {"time": 0, "action": "standing", "confidence": 0.90}
      ]
    }
  },
  "id_time_actions": [
    {"id": "id_1", "time": 0, "action": "standing", "confidence": 0.90}
  ]
}
```

## 🔧 Configuration

### ByteTrack Parameters (`modules/bytetrack_tuned.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_high_thresh` | 0.5 | First-stage matching threshold |
| `track_low_thresh` | 0.1 | Second-stage low-confidence matching |
| `new_track_thresh` | 0.6 | New track creation threshold |
| `track_buffer` | 60 | Lost track retention frames |
| `match_thresh` | 0.8 | IoU-based association threshold |

## 📚 Module Documentation

For detailed module documentation, see the `docs/` directory:

- [detector.md](detector.md) - Detection and tracking module
- [recognizer.md](recognizer.md) - Action recognition module
- [exporter.md](exporter.md) - Label data export module
- [dataset.md](dataset.md) - Dataset utilities
- [bytetrack_config.md](bytetrack_config.md) - ByteTrack configuration
- [logger.md](logger.md) - Logging utility

## 📦 Dependencies

- Python >= 3.8
- PyTorch >= 2.1.0
- ultralytics >= 8.0.0
- transformers >= 4.38.0
- OpenCV
- Rich (logging)
- Qwen-VL-Utils

## 🔍 Performance Considerations

- **GPU Memory**: Qwen3-VL-8B requires approximately 16GB VRAM
- **Processing Time**: ~5-7 minutes for 1-minute video (GPU, iterations=5)
- **Minimum Tube Length**: Tracks under 30 frames are filtered out

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
