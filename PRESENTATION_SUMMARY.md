# Smart Notes Generator - Presentation Summary

**Date:** January 27, 2026  
**Project:** Automated Lecture Note Generation from Video Lectures  
**Status:** Model Training & Deduplication Complete ✅

---

## 📊 Project Overview

Developed an XGBoost-based machine learning system to automatically detect slide transitions in educational lecture videos and prepare them for multimodal AI note generation.

### Dataset
- **19 lecture videos** (Chemistry, Physics, Mathematics, Computer Science, Algorithms, Databases)
- **110,249 frames** analyzed across all videos
- **25 visual features** extracted per frame (edge detection, histogram analysis, motion detection, color analysis)

---

## 🎯 Model Performance

### Training Results
- **Model:** XGBoost with temporal smoothing (NMS-style filtering)
- **Accuracy:** 91% on test set
- **Recall:** 86.36% (captures 86% of actual transitions)
- **Threshold:** 0.10 (optimized for high recall)

### Deduplication Results
- **Raw model predictions:** 3,588 transitions detected
- **After SSIM deduplication:** 331 unique transitions
- **Deduplication rate:** 90.8% reduction
- **Method:** Sequential SSIM comparison (threshold: 0.92)

---

## 📁 Deliverables for Presentation

### 1. Visual Previews (data/transition_previews/)

Each video has TWO HTML preview files:

#### A. Raw Transitions (`<video_id>_transitions.html`)
- Shows ALL transitions detected by the model
- Includes duplicate detections from consecutive frames
- Example: toc_1 shows 88 raw detections

#### B. Deduplicated Transitions (`<video_id>_deduplicated_transitions.html`)
- Shows FINAL unique transitions after SSIM filtering
- Blank slides marked with 🚫 and orange border
- Includes SSIM scores, edge counts, and audio duration
- Example: toc_1 shows 9 final transitions (5 blank, 4 content)

**How to present:** Open both HTML files side-by-side to demonstrate the effectiveness of deduplication.

### 2. Transition Data (data/transition_previews/deduplicated_transitions.json)

Complete structured data with:
- Exact timestamp for each transition
- SSIM similarity scores
- Blank slide flags (edge_count < 0.1)
- Audio window segments (start/end timestamps)

---

## 📈 Per-Video Statistics

| Video ID | Total Transitions | Blank Slides | Content Slides |
|----------|------------------|--------------|----------------|
| algo_1 | 10 | 0 | 10 |
| algorithms_14_hindi | 6 | 2 | 4 |
| chemistry_01_english | 33 | 0 | 33 |
| chemistry_04_english | 31 | 0 | 31 |
| chemistry_08_hindi | 31 | 0 | 31 |
| chemistry_09_hindi | 26 | 0 | 26 |
| chemistry_10_english | 6 | 0 | 6 |
| cn_1 | 17 | 3 | 14 |
| computer_networks_13_hindi | 8 | 0 | 8 |
| database_11_hindi | 8 | 0 | 8 |
| database_12_hindi | 5 | 0 | 5 |
| database_13_hindi | 24 | 3 | 21 |
| db_1 | 8 | 2 | 6 |
| mathematics_03_english | 32 | 0 | 32 |
| mathematics_05_hindi | 12 | 2 | 10 |
| mathematics_07_hindi | 6 | 0 | 6 |
| physics_01_english | 25 | 15 | 10 |
| physics_05_english | 34 | 2 | 32 |
| toc_1 | 9 | 5 | 4 |
| **TOTAL** | **331** | **34** | **297** |

---

## 🔬 Technical Highlights

### 1. Smart Deduplication Algorithm
Implemented a sophisticated 5-point deduplication system:

1. **Sequential Comparison:** Compare each transition to the previous accepted one using SSIM
2. **Rapid-Fire Detection:** Keep transitions < 2s apart if visually different (SSIM < 0.85)
3. **Blank Slide Suppression:** Flag slides with low edge count (< 0.1) but still include them
4. **Audio Segmentation:** Calculate precise audio windows between transitions
5. **End-of-Video Logic:** Force final capture at video end for completeness

### 2. Performance Optimization
- **Min-gap filtering:** Skip transitions within 5 seconds of previous
- **Reduced lookback window:** 5 seconds instead of 10 (2x speedup)
- **30 FPS extraction:** Native video frame rate for accuracy
- **Processing time:** ~2 minutes per video

### 3. Model Predictions vs Ground Truth
- **Initial confusion:** System accidentally used manual ground truth timestamps
- **Resolution:** Fixed to use XGBoost model predictions (`smoothed_prediction` column)
- **Validation:** Verified on 3 videos - model predictions closely match manual labels
- **Conclusion:** Model is performing exceptionally well (86% recall validated)

---

## 📂 Folder Structure for Presentation

```
data/
├── transition_previews/
│   ├── deduplicated_transitions.json ← Final output data
│   ├── toc_1_transitions.html ← Raw detections
│   ├── toc_1_deduplicated_transitions.html ← Final deduplicated
│   ├── algo_1_transitions.html
│   ├── algo_1_deduplicated_transitions.html
│   └── ... (19 videos × 2 files each)
│
└── videos/ ← Original lecture videos
```

---

## 🎯 Next Steps

### Phase 1: High-Resolution Frame Extraction
- Extract 331 PNG images at full resolution (1920×1080)
- Organize by video: `data/final_frames/<video_id>/<video_id>_slide_<number>.png`

### Phase 2: Audio Extraction
- Extract 331 audio segments using FFmpeg
- Use audio window timestamps from deduplicated_transitions.json
- Format: MP3, organized per video

### Phase 3: Multimodal AI Integration
- Feed slide images + audio + metadata to GPT-4 Vision or Gemini
- Generate structured lecture notes automatically
- Include: key concepts, explanations, equations, diagrams

### Phase 4: Model Refinement
- Improve blank slide detection accuracy
- Add teacher presence detection
- Fine-tune SSIM thresholds based on subject matter

---

## 💡 Key Achievements

✅ **19 videos processed** through complete ML pipeline  
✅ **86.36% recall** - model catches vast majority of transitions  
✅ **90.8% deduplication** - reduces 3,588 → 331 unique transitions  
✅ **Smart blank detection** - identifies 34 empty slides automatically  
✅ **Production-ready data** - JSON + HTML previews for next phase  

---

## 📞 Questions for Ma'am

1. **Blank Slide Handling:** Should we exclude blank slides from note generation or include them with special marking?
2. **Audio Segmentation:** Current approach uses audio between transitions - is this the right granularity?
3. **Model Performance:** 86% recall means ~14% transitions missed - acceptable threshold?
4. **Physics Videos:** physics_01_english has 60% blank slides - investigate video quality?
5. **Next Phase Priority:** Should we proceed with high-res extraction or refine the model first?

---

**Prepared by:** Smart Notes Generator Team  
**Contact:** [Your contact information]  
**Repository:** `d:\College_Life\projects\smart notes generator - trail 3`
