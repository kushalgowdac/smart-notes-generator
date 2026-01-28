# Smart Notes Generator - Video Feature Extraction

## 📖 Overview
This project extracts specialized features from lecture videos (like Physics Wallah videos) to detect transitions, slide changes, and important moments for automated note generation.

## 🎯 Features Extracted

### 1. **Teacher Detection (Enhanced)**
- **Black Pixel Ratio**: Detects teacher's black t-shirt (RGB < 50)
- **Skin Pixel Ratio**: HSV-based skin detection for face/hands
  - HSV Range: Lower [0, 20, 70], Upper [20, 255, 255]
- **Teacher Presence**: Combined metric (black + skin pixels)
  - **Key Insight**: High teacher_presence + Low SSIM = Teacher motion, NOT transition!

### 2. **Global Differences**
- **SSIM (Structural Similarity Index)**: Measures frame similarity
- **MSE (Mean Squared Error)**: Quantifies pixel-level changes
- **Histogram Correlation**: Tracks color distribution changes

### 3. **Edge Analysis**
- Canny Edge Detection for content changes
- Edge Count & Edge Change Rate
- Edge Decay Velocity (distinguishes instant jumps vs gradual erasing)

### 4. **Tri-Zonal ROI Analysis**
- Bottom 20% of frame divided into 3 zones (Left, Center, Right)
- Local SSIM and Edge Density per zone
- Detects UI interactions and regional changes

### 5. **Temporal Memory (Sliding Window)**
- 10-frame (2-second) rolling window
- Rolling Mean, Std Dev, and Max for SSIM and Edge metrics
- Temporal pattern detection

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Videos placed in `data/videos/` folder

### Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Place your videos** in the `data/videos/` folder

3. **Run the extractor**:
```bash
# Process all videos in data/videos/
python src/video_feature_extractor.py

# Or specify custom folders
python src/video_feature_extractor.py --input data/videos --output data/output

# Process a single video
python src/video_feature_extractor.py --single "path/to/video.mp4"

# Change processing FPS (default is 5)
python src/video_feature_extractor.py --fps 10
```

## 📊 Output

Each video generates a separate CSV file in `data/output/` with the following structure:

| Column | Description |
|--------|-------------|
| `video_id` | Video filename (without extension) |
| `frame_index` | Sequential frame number (at 5 FPS) |
| `timestamp_seconds` | Time in video (seconds) |
| `black_pixel_ratio` | Ratio of near-black pixels (0-1) |
| `skin_pixel_ratio` | Ratio of skin-colored pixels (0-1) **NEW** |
| `teacher_presence` | Combined black + skin ratio (0-2) **NEW** |
| `global_ssim` | Global SSIM between consecutive frames (0-1) |
| `global_mse` | Mean Squared Error (0-1, normalized) |
| `histogram_correlation` | Color histogram correlation (0-1) |
| `edge_count` | Total edge pixels detected (0-1, normalized) |
| `edge_change_rate` | Change in edge count (0-1, normalized) |
| `zone_left_ssim` | Left zone SSIM (0-1) |
| `zone_left_edge_density` | Left zone edge density (0-1) |
| `zone_center_ssim` | Center zone SSIM (0-1) |
| `zone_center_edge_density` | Center zone edge density (0-1) |
| `zone_right_ssim` | Right zone SSIM (0-1) |
| `zone_right_edge_density` | Right zone edge density (0-1) |
| `ssim_rolling_mean` | 2-second rolling mean of SSIM (0-1) |
| `ssim_rolling_std` | 2-second rolling std dev of SSIM (0-1) |
| `ssim_rolling_max` | 2-second rolling max of SSIM (0-1) |
| `edge_rolling_mean` | 2-second rolling mean of edges (0-1) |
| `edge_rolling_std` | 2-second rolling std dev of edges (0-1) |
| `edge_rolling_max` | 2-second rolling max of edges (0-1) |
| `edge_decay_velocity` | Slope of edge count (0-1, normalized) |
| `label` | Ground truth label (0 = no transition) |

**Total: 25 features (all Min-Max normalized 0-1 per video, except teacher_presence which can be 0-2).**

## 🔧 Command-Line Options

```bash
python src/video_feature_extractor.py [OPTIONS]

Options:
  -i, --input PATH    Input folder with videos (default: data/videos)
  -o, --output PATH   Output folder for CSVs (default: data/output)
  -f, --fps INT       Target FPS for processing (default: 5)
  -s, --single PATH   Process single video file
  -h, --help          Show help message
```

## 📁 Project Structure

```
smart notes generator - trail 3/
├── src/
│   └── video_feature_extractor.py    # Main extraction script
├── data/
│   ├── videos/                        # Input videos go here
│   └── output/                        # Output CSV files
├── logs/
│   └── feature_extraction.log         # Processing logs
├── docs/
│   └── README.md                      # This file
├── requirements.txt                   # Python dependencies
└── PROJECT_HISTORY_MASTER.md          # Master documentation
```

## 🎬 Example Usage

### Process All Videos (New Features)
```bash
# Default settings (5 FPS, data/videos/ folder)
python src/video_feature_extractor.py
```

### Re-process Existing Videos with New Features
```bash
# This will backup old CSVs and re-process with skin detection
python src/reprocess_videos.py

# Skip backup (overwrite old files)
python src/reprocess_videos.py --no-backup

# Custom backup location
python src/reprocess_videos.py --backup-dir "data/backups"
```

### Process Single Video
```bash
python src/video_feature_extractor.py --single "data/videos/lecture_01.mp4"
```

### Custom Settings
```bash
python src/video_feature_extractor.py --input "D:/my_videos" --output "D:/results" --fps 10
```

## 📈 Performance Notes

- **Processing Speed**: ~5-10x faster than real-time (5 FPS sampling)
- **Memory Usage**: ~500MB - 2GB depending on video resolution
- **Disk Space**: CSV files are ~100-500KB per minute of video

## 🛠️ Troubleshooting

### Video Won't Process
- Check video codec compatibility (MP4/H.264 recommended)
- Ensure video file isn't corrupted
- Check logs in `logs/feature_extraction.log`

### Out of Memory
- Reduce FPS (use `--fps 3` or lower)
- Process videos one at a time using `--single`

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Next Steps

After feature extraction:
1. Manually label transitions in CSV files (set `label` column to 1)
2. Merge multiple CSV files for training
3. Train a machine learning model for automatic transition detection
4. Use the model to auto-generate timestamps and notes

## 🤝 Contributing

For questions or issues, check the logs or update the PROJECT_HISTORY_MASTER.md file with your findings.

---

**Version**: 1.0.0  
**Last Updated**: January 25, 2026  
**Storage Location**: D:\College_Life\projects\smart notes generator - trail 3
