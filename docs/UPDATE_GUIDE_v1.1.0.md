# Smart Notes Generator - Version 1.1.0 Update Guide

## 🆕 What's New in Version 1.1.0

### New Features Added
1. **skin_pixel_ratio** - HSV-based skin detection
   - Detects teacher's face and hands
   - Range: HSV [0,20,70] to [20,255,255]
   
2. **teacher_presence** - Combined teacher detection
   - Formula: `black_pixel_ratio + skin_pixel_ratio`
   - Helps distinguish teacher motion from slide transitions

### Why These Features?

**Problem:** The model was confusing teacher movement with slide transitions.

**Solution:** 
- When `global_ssim` is low + `teacher_presence` is high → **Teacher Motion** (NOT a transition)
- When `global_ssim` is low + `teacher_presence` is low → **Slide Change** (TRANSITION!)

## 📊 CSV Structure Changes

### Old CSV (23 features):
```
video_id, frame_index, timestamp_seconds,
black_pixel_ratio,
global_ssim, global_mse, histogram_correlation,
edge_count, edge_change_rate,
zone_left_ssim, zone_left_edge_density,
zone_center_ssim, zone_center_edge_density,
zone_right_ssim, zone_right_edge_density,
ssim_rolling_mean, ssim_rolling_std, ssim_rolling_max,
edge_rolling_mean, edge_rolling_std, edge_rolling_max,
edge_decay_velocity,
label
```

### New CSV (25 features):
```
video_id, frame_index, timestamp_seconds,
black_pixel_ratio,
skin_pixel_ratio,           ← NEW!
teacher_presence,            ← NEW!
global_ssim, global_mse, histogram_correlation,
edge_count, edge_change_rate,
zone_left_ssim, zone_left_edge_density,
zone_center_ssim, zone_center_edge_density,
zone_right_ssim, zone_right_edge_density,
ssim_rolling_mean, ssim_rolling_std, ssim_rolling_max,
edge_rolling_mean, edge_rolling_std, edge_rolling_max,
edge_decay_velocity,
label
```

## 🚀 How to Update Your Data

### Option 1: Re-process All Videos (Recommended)
```bash
# Automatically backs up old CSVs to data/output_backup/
python src/reprocess_videos.py
```

This will:
✅ Backup your old 23-feature CSVs
✅ Re-process all 19 videos with new features
✅ Generate updated 25-feature CSVs

**Time:** ~15-17 hours for all 19 videos

### Option 2: Process Only New Videos
```bash
# Just use the updated script for new videos
python src/video_feature_extractor.py --single "data/videos/new_video.mp4"
```

### Option 3: Keep Old Files, Process Subset
```bash
# Move videos you want to reprocess to a temp folder
python src/reprocess_videos.py --input "data/videos_subset" --no-backup
```

## 📁 File Locations

- **Updated Script:** `src/video_feature_extractor.py`
- **Re-processing Utility:** `src/reprocess_videos.py`
- **Old CSVs (after backup):** `data/output_backup/[timestamp]/`
- **New CSVs:** `data/output/`

## 🎯 ML Model Benefits

With these new features, your ML model will:
1. **Better distinguish** teacher motion from slide changes
2. **Reduce false positives** (teacher walking ≠ transition)
3. **Improve accuracy** by ~10-15% (estimated)

## 🔍 Quick Check

To verify the new features are working:

```python
import pandas as pd

# Load a new CSV
df = pd.read_csv('data/output/algorithms_14_hindi_features.csv')

# Check columns (should be 25 now)
print(f"Total columns: {len(df.columns)}")
print(f"New features present: {{'skin_pixel_ratio', 'teacher_presence'}.issubset(df.columns)}")

# View new features
print(df[['timestamp_seconds', 'black_pixel_ratio', 'skin_pixel_ratio', 'teacher_presence']].head())
```

Expected output:
```
Total columns: 25
New features present: True
```

## ⚠️ Important Notes

1. **Old CSVs won't work** with models trained on new 25-feature data
2. **New CSVs won't work** with models trained on old 23-feature data
3. **Always re-train** your ML model after updating features
4. **Backup is automatic** unless you use `--no-backup` flag

## 📞 Need Help?

Check the logs:
```bash
# View processing logs
cat logs/feature_extraction.log

# Windows
type logs\feature_extraction.log
```

---

**Version:** 1.1.0  
**Updated:** January 26, 2026  
**Location:** D:\College_Life\projects\smart notes generator - trail 3
