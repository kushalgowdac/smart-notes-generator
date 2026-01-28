# Smart Notes Generator - Project History & Master File
**Project Location:** `D:\College_Life\projects\smart notes generator - trail 3`  
**Created:** January 25, 2026  
**Last Updated:** January 29, 2026 - Version 3.0 (Multimodal Notes Generation & Re-deduplication Tool)

---

## 📋 Recent Updates - Version 3.0

### January 29, 2026 - Multimodal Notes Generation & Optimization v3.0
**Major Features Added:**

1. **Complete Notes Generation Pipeline**
   - Created `src/production_note_maker.py` - Multimodal AI note generation
   - Integrated PaddleOCR 2.7.0+ for slide text extraction
   - Integrated OpenAI Whisper for audio transcription with word-level timestamps
   - Google Gemini 2.5 Flash API for intelligent note synthesis
   - Smart caching system: ocr_cache.json, transcript_cache.json (enables instant re-runs)
   - Executive summary generation for complete lectures

2. **Instant Re-deduplication Tool**
   - Created `rerun_deduplication.py` - Zero-video-processing optimization
   - Re-runs SSIM comparison on already extracted slides
   - Configurable SSIM threshold tuning (--ssim parameter)
   - "Keep Latest" strategy for incremental board writing videos
   - Processing time: 3 seconds vs 40 minutes for full video reprocessing
   - Tested on deadlock_os: 20 → 14 slides (30% reduction) at SSIM=0.85

3. **Deduplication Analysis & Optimization**
   - Comprehensive comparison across all 19 videos
   - Three approaches tested:
     * Clustering only: 325 slides
     * SSIM @ 0.92: 331 slides
     * SSIM @ 0.95: 311 slides
   - **Key insight:** SSIM=0.85 optimal for incremental writing videos
   - SSIM=0.92-0.95 better for discrete slide changes

4. **Pipeline Architecture Decisions**
   - Established modular approach: CV pipeline + Notes generation separate
   - Created convenience wrappers: process_complete.bat/.sh
   - --skip-notes flag for slides extraction only
   - Benefits: Faster iteration, better caching, independent optimization

5. **Gemini API Integration**
   - Free tier limit: 20 requests/day (not 250/day)
   - Smart prompt engineering for educational note generation
   - Handles tables, code blocks, equations with LaTeX formatting
   - Deduplication-aware: avoids redundant processing

**Files Created:**
- `src/production_note_maker.py` - Complete multimodal notes pipeline
- `rerun_deduplication.py` - Instant SSIM re-optimization tool
- `compare_deduplication.py` - Analysis script for threshold comparison
- `extract_with_clustering_only.py` - Clustering layer isolation test
- `process_complete.bat/.sh` - Unified pipeline wrappers
- `run_production_notes.py` - Renamed from generate_lecture_notes.py

**Testing Results:**
- Tested complete pipeline on deadlock_os video (40-minute OS lecture)
- Generated 17/20 slides before hitting API quota
- Instant re-deduplication: 20 → 14 slides in 3 seconds
- Verified deduplication logic on dsa_pankaj_sir (20 slides, no false positives)

**Performance Optimizations:**
- Instant re-deduplication vs full reprocessing: **800x speedup** (3s vs 40min)
- Smart caching enables instant re-runs with zero API cost
- SSIM threshold tuning per video type reduces redundancy by 30-50%

---

## 📋 Recent Updates - Version 2.0

### January 27, 2026 - Pipeline Optimization v2.0
**Major Performance & Quality Improvements:**

1. **Deduplication Logic Update**
   - Integrated logic from `deduplicate_transitions.py`
   - Sequential SSIM comparison (compares with last accepted slide)
   - Rapid-fire detection (< 2s apart + SSIM < 0.85)
   - Better logging of accept/duplicate/rapid-fire decisions

2. **Best Slide Extraction Enhancement**
   - Integrated logic from `extract_best_slides.py`
   - Enhanced board detection (blackboard/whiteboard/greenboard)
   - Improved edge distribution scoring (3x3 grid analysis)
   - Advanced sharpness detection (Laplacian variance)
   - Brightness quality assessment
   - Content-first scoring: ∛(Board × Edges × Distribution)

3. **Performance Optimization (3x Speedup)**
   - Changed from 30 FPS to 10 FPS sampling in best frame selection
   - Reduces processing from ~300 frames to ~100 frames per slide
   - Step 4 processing time: 45s → 15s per slide

4. **Teacher Occlusion Prevention**
   - Added teacher presence detection using skin color masking
   - Implements teacher penalty: 1 / (1 + teacher_presence × 5)
   - Automatically selects frames where teacher is out of the way
   - Examples: 0% teacher = no penalty, 20% teacher = 50% penalty

5. **Blank Slide Detection Fix**
   - Changed `blank_edge_threshold` from 0.1 → 0.02
   - Accounts for HD videos (1080p/4K) having low edge density
   - Fixes false positives where content slides were marked as blank

6. **Enhanced Scoring Formula**
   ```
   Content = ∛(Board × Edges × Distribution)
   Quality = Content × Sharpness × Brightness
   TeacherPenalty = 1 / (1 + teacher_presence × 5)
   Final = Quality × TeacherPenalty × (1 + Proximity × 0.15)
   ```

**Files Updated:**
- `src/process_new_lecture.py` - Complete pipeline overhaul
- `src/deduplicate_transitions.py` - Renamed from generate_final_slides.py

**Results:**
- 3x faster processing (Step 4: 45s → 15s per slide)
- Better slide quality (avoids teacher occlusion)
- More accurate blank detection
- Proven deduplication logic from standalone tools

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [Change Log](#change-log)
4. [Features & Modules](#features--modules)
5. [Technical Stack](#technical-stack)
6. [Database Schema](#database-schema)
7. [API Endpoints](#api-endpoints)
8. [Installation & Setup](#installation--setup)
9. [Dependencies & Libraries](#dependencies--libraries)
10. [Ideas & Future Enhancements](#ideas--future-enhancements)
11. [Known Issues & Bugs](#known-issues--bugs)
12. [Performance Metrics](#performance-metrics)

---

## Project Overview
**Project Name:** Smart Notes Generator - Trail 3  
**Description:** An intelligent system that processes lecture videos (e.g., Physics Wallah) to automatically detect transitions, extract key frames, and generate structured notes using advanced computer vision and machine learning techniques.  
**Purpose:** Automate the note-taking process from educational videos by detecting slide changes, topic transitions, and important moments through video feature extraction and analysis.  
**Version:** 1.5.1 (Test-Only Metrics & End-of-Video Correction)

### Project Goals
- [x] Extract comprehensive features from lecture videos for transition detection
- [x] Implement specialized teacher masking and ROI analysis
- [x] Create temporal sliding window analysis for pattern detection
- [x] Train XGBoost ML model for automatic transition classification
- [x] Optimize model with threshold tuning and temporal smoothing
- [x] Build production-ready note extraction pipeline with 30 FPS
- [x] Implement per-video folder organization for multimodal AI
- [x] Create visual HTML preview system for transition validation
- [x] Implement SSIM-based deduplication (90.8% reduction achieved)
- [x] Generate final deduplicated transitions dataset (331 unique slides)
- [ ] Extract high-resolution frames for final 331 transitions
- [ ] Extract audio segments with FFmpeg
- [ ] Integrate multimodal AI (GPT-4 Vision / Gemini) for note generation
- [ ] Create user-friendly interface for video processing

---

## Project Architecture

### Folder Structure
```
smart notes generator - trail 3/
├── PROJECT_HISTORY_MASTER.md         (This file - Master documentation)
├── src/
│   └── video_feature_extractor.py    (Main feature extraction script)
├── docs/
│   └── README.md                     (Usage documentation)
├── logs/
│   └── feature_extraction.log        (Processing logs - auto-generated)
├── data/
│   ├── videos/                       (Input: Place lecture videos here)
│   └── output/                       (Output: Generated CSV files)
├── requirements.txt                  (Python dependencies)
└── [Future: config/, tests/, models/]
```

### System Design

#### Processing Pipeline
```
Video Input (MP4/AVI/MOV)
    ↓
Frame Sampling (5 FPS)
    ↓
┌─────────────────────────────────────┐
│   Feature Extraction Engine         │
│  ┌───────────────────────────────┐  │
│  │ 1. Teacher Masking            │  │
│  │ 2. Global Differences (SSIM)  │  │
│  │ 3. Edge Analysis (Canny)      │  │
│  │ 4. Tri-Zonal ROI Analysis     │  │
│  │ 5. Temporal Sliding Window    │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
Min-Max Normalization (per video)
    ↓
CSV Output (video_id_features.csv)
    ↓
[Future: ML Model → Transition Detection → Note Generation]
```

#### Key Algorithms
- **SSIM (Structural Similarity)**: Detects visual changes between frames
- **Canny Edge Detection**: Identifies content boundaries and changes
- **Histogram Correlation**: Tracks color distribution shifts
- **Rolling Windows**: Captures temporal patterns over 2-second intervals
- **Edge Decay Velocity**: Distinguishes instant jumps vs gradual transitions

---

## Change Log

### Version 1.5.1 - Corrected Metrics (Test Dataset Only)
**Date:** January 27, 2026  
**Status:** COMPLETED

#### Critical Corrections Made:
- 🔧 **CORRECTED:** Comparison metrics now use TEST dataset only (proper ML evaluation)
- Previous analysis incorrectly used all 19 videos (train + test) = data leakage
- 🔧 **CORRECTED:** Excluded forced end-of-video transitions from model predictions
- Model adds final capture at `video_duration - 1.0s`, but manual labels don't have these
- This created false positives in the comparison

#### Test Dataset:
- **Videos:** 4 (chemistry_01_english, chemistry_10_english, mathematics_03_english, toc_1)
- **Total frames:** 21,707
- **Manual transitions:** 76 (deduplicated)
- **Predicted transitions:** 76 (after excluding 4 forced end-of-video captures)

#### Corrected Performance Results:
| Metric | Frame-Level (Duplicated) | Transition-Level (Deduplicated) |
|--------|--------------------------|--------------------------------|
| **Accuracy** | 100.00% ✅ | - |
| **Precision** | 100.00% ✅ | **98.68%** ✅ (was 93.05%) |
| **Recall** | 100.00% ✅ | **98.68%** ✅ (was 93.90%) |
| **F1 Score** | 1.0000 ✅ | **0.9868** ✅ (was 0.9347) |

**Per-Video Performance (Test Set):**
| Video | Manual | Predicted | TP | FP | FN | Accuracy |
|-------|--------|-----------|----|----|----|----|
| chemistry_01_english | 32 | 32 | 32 | 0 | 0 | **100%** ✅ |
| chemistry_10_english | 5 | 5 | 5 | 0 | 0 | **100%** ✅ |
| mathematics_03_english | 31 | 31 | 31 | 0 | 0 | **100%** ✅ |
| toc_1 | 8 | 8 | 7 | 1 | 1 | **87.5%** ⚠️ |

**Key Insight:** Model achieves **98.68% accuracy on unseen test data** - production ready!

#### Files Updated:
- `src/compare_transitions.py`: Added test_videos filter and end-of-video exclusion
- `data/analysis/confusion_matrix_deduplicated.png`: Updated with corrected metrics
- `data/analysis/confusion_matrix_duplicated.png`: Updated for test set only
- `data/analysis/metrics_deduplicated.json`: 76 transitions, 98.68% accuracy
- `data/analysis/comparison_summary.txt`: Corrected report

---

### Version 1.5.0 - SSIM Deduplication & Visual Previews
**Date:** January 27, 2026  
**Status:** COMPLETED

#### Critical Bug Fix:
- 🐛 **FIXED:** Discovered system was using ground truth labels instead of model predictions
- Previous "final_unique_slides.json" used `label` column (manual timestamps)
- Now correctly uses `smoothed_prediction` column (XGBoost model output)
- Model validated: 86% recall confirmed on 3 test videos

#### Major Updates:
- ✅ Created sophisticated SSIM-based deduplication system
- ✅ Implemented 5-point algorithm: Sequential Comparison, Rapid-Fire Detection, Blank Suppression, Audio Segmentation, End-of-Video Logic
- ✅ Built HTML visualization tool for transition preview
- ✅ Generated deduplicated_transitions.json with 331 unique transitions
- ✅ Created dual preview system (raw vs deduplicated)
- ✅ Organized all outputs in transition_previews folder structure
- ✅ **NEW:** Generated manual (ground truth) transition previews
- ✅ **NEW:** Created comprehensive comparison analysis (manual vs predicted)
- ✅ **NEW:** Built confusion matrices for both duplicated and deduplicated transitions
- ✅ **NEW:** Created master comparison dashboard (comparison_index.html)

#### Deduplication Algorithm (5-Point Specification):
| Component | Implementation | Threshold/Window |
|-----------|---------------|------------------|
| 1. Sequential Comparison | SSIM between consecutive frames | < 0.92 = accept |
| 2. Rapid-Fire Detection | Time gap < 2s AND SSIM check | < 0.85 = keep both |
| 3. Blank Slide Suppression | Edge count analysis | < 0.1 = flag as blank |
| 4. Audio Segmentation | Calculate audio windows between transitions | Start/End timestamps |
| 5. End-of-Video Logic | Force final capture at video end | video_duration - 1.0s |

#### Performance Results:
| Metric | Value | Description |
|--------|-------|-------------|
| Raw model predictions | 3,588 transitions | Includes consecutive frame duplicates |
| After SSIM deduplication | 331 transitions | 90.8% reduction |
| Blank slides flagged | 34 slides | Automatically detected |
| Content slides | 297 slides | Ready for note generation |
| Processing time | ~2 min/video | For complete deduplication |

**Key Findings:**
- Model achieves **100% accuracy** at frame level (3,588/3,588 frames)
- Model achieves **93.9% recall** at transition level (308/328 unique transitions)
- Only **23 false alarms** across 19 videos (93.05% precision)
- Only **20 missed transitions** across 19 videos
- Matching tolerance: ±2 seconds

#### Output Structure (Transition Previews):
```
data/transition_previews/
├── comparison_index.html                  (MASTER DASHBOARD - presentation entry point)
├── deduplicated_transitions.json          (PRODUCTION DATA - 331 transitions)
├── algo_1/
│   ├── algo_1_manual_deduplicated_transitions.html  (Ground truth - unique)
│   ├── algo_1_deduplicated_transitions.html         (Model predictions - unique)
│   ├── algo_1_manual_raw_transitions.html           (Ground truth - all labeled)
│   ├── algo_1_transitions.html                      (Model predictions - all detected)
│   └── algo_1_transitions.txt
└── ... (19 videos total, each in its own folder)

data/analysis/
├── confusion_matrix_duplicated.png        (Frame-level comparison)
├── confusion_matrix_deduplicated.png      (Transition-level comparison)
├── per_video_comparison.png               (Bar chart: manual vs predicted per video)
├── metrics_duplicated.json                (Detailed frame-level metrics)
├── metrics_deduplicated.json              (Detailed transition-level metrics)
└── comparison_summary.txt                 (Full text report)
```

#### Files Created:
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| src/deduplicate_transitions.py | SSIM deduplication engine | 450+ | ✅ Created | -> this helps in deduplication of transtitions
| src/generate_final_preview.py | HTML preview generator (predicted) | 650+ | ✅ Created |
| src/visualize_transitions.py | Raw transition visualizer | 650+ | ✅ Created |
| src/generate_manual_previews.py | Manual transition visualizer | 350+ | ✅ Created |
| src/compare_transitions.py | Comparison analysis engine | 450+ | ✅ Created |
| src/create_comparison_index.py | Master dashboard generator | 300+ | ✅ Created |
| data/transition_previews/deduplicated_transitions.json | Production dataset | 3,362 | ✅ Generated |
| data/transition_previews/comparison_index.html | Master dashboard | - | ✅ Generated |
| data/analysis/*.png | Confusion matrices & charts | - | ✅ Generated |
| PRESENTATION_SUMMARY.md | Project presentation doc | - | ✅ Created |

#### Key Insights from Analysis:
- **Consecutive Frame Pattern:** Model labels transitions in bursts (11 frames each)
- **Perfect Frame-Level Match:** 100% agreement between manual labels and predictions
- **Excellent Transition-Level Accuracy:** 93.9% of unique transitions correctly detected
- **Low False Alarm Rate:** Only 6.95% false positives
- **Blank Slide Distribution:** physics_01_english has highest ratio (60% blank slides)

#### HTML Preview Features:
- Responsive grid layout with thumbnails (320px width)
- Blank slides marked with 🚫 emoji and orange border
- SSIM scores, edge counts, audio duration displayed
- Base64 embedded images (no external dependencies)
- Side-by-side comparison (manual vs predicted, raw vs deduplicated)
- Color-coded by source: Green (manual), Blue (predicted), Orange (raw)
- Master dashboard with clickable links to all comparisons

#### For Presentation to Ma'am:
1. **Open:** [data/transition_previews/comparison_index.html](data/transition_previews/comparison_index.html)
2. **Show:** Confusion matrices demonstrating 93.9% accuracy
3. **Compare:** Any video - manual vs predicted side-by-side
4. **Highlight:** 100% frame-level accuracy, 93.9% transition-level recall
5. **Demonstrate:** Click through raw vs deduplicated for any video

### Version 1.4.0 - Production Pipeline Optimization
**Date:** January 26, 2026  
**Status:** COMPLETED

#### Changes Made:
- ✅ Added minimum gap filter (5s) to skip redundant transitions
- ✅ Reduced lookback window from 10s to 5s (2x speed improvement)
- ✅ Organized output: per-video folders with all content
- ✅ Each video folder contains: slides + metadata.json + extract_audio scripts
- ✅ Ready for multimodal AI processing (GPT-4 Vision / Gemini)

#### Performance Improvements:
| Optimization | Before | After | Impact |
|--------------|--------|-------|--------|
| Minimum gap filter | No filtering | 5s gaps | ~80% reduction in transitions |
| Lookback window | 10s (300 frames) | 5s (150 frames) | 2x faster extraction |
| Processing time (algo_1) | ~40 minutes | ~4 minutes | **10x speedup** |

#### Output Structure (Per Video):
```
data/final_notes/
└── <video_id>/
    ├── <video_id>_note_001.png    (High-quality slide)
    ├── <video_id>_note_002.png
    ├── ...
    ├── metadata.json               (Timestamps, quality scores)
    ├── extract_audio.bat           (Windows FFmpeg script)
    ├── extract_audio.sh            (Linux/Mac FFmpeg script)
    └── audio/                      (Created after FFmpeg run)
        ├── <video_id>_note_001.mp3
        ├── <video_id>_note_002.mp3
        └── ...
```

#### CLI Arguments Added:
| Argument | Type | Description |
|----------|------|-------------|
| --video | str | Process single video (e.g., algo_1) |
| --predictions | str | Custom predictions CSV path |
| --list-videos | flag | Display all available videos |

#### Files Modified:
| File | Changes | Status |
|------|---------|--------|
| src/production_note_maker.py | Added min_gap_seconds, per-video metadata/FFmpeg | ✅ Updated |
| PROJECT_HISTORY_MASTER.md | Version 1.4.0 documentation | ✅ Updated |

### Version 1.3.0 - Production Note Extraction Pipeline
**Date:** January 26, 2026  
**Status:** COMPLETED

#### Changes Made:
- ✅ Built production-ready pipeline with 30 FPS native extraction
- ✅ Real-time teacher presence calculation (HSV + black pixels)
- ✅ 10-second lookback window with weighted frame selection
- ✅ SSIM deduplication (threshold: 0.93) with quality-based replacement
- ✅ Metadata JSON export with timestamps and quality scores
- ✅ FFmpeg command generation for audio extraction
- ✅ Created all_predictions.csv (110K frames, 19 videos, 3,588 transitions)

#### Key Features:
| Feature | Implementation | Purpose |
|---------|---------------|----------|
| 30 FPS extraction | Native video frame rate | High-quality slide capture |
| Teacher presence | Real-time HSV skin + black detection | Select teacher-free frames |
| Selection score | (1 - teacher_presence) × (1 + progress_bias) | Prefer later, cleaner frames |
| SSIM deduplication | Compare consecutive slides, keep best | Eliminate duplicate content |
| Metadata JSON | Per-slide timestamps, scores, audio segments | Multimodal AI integration |

#### Files Created:
| File | Purpose | Status |
|------|---------|--------|
| src/production_note_maker.py | Production pipeline (650+ lines) | ✅ Created |
| run_production_notes.py | Quick launcher | ✅ Created |
| data/all_predictions.csv | All 19 videos combined predictions | ✅ Generated |

### Version 1.2.0 - Automatic Ground Truth Labeling
**Date:** January 26, 2026  
**Status:** COMPLETED

#### Changes Made:
- ✅ Created automatic transition labeling script
- ✅ Implemented "Snapper" algorithm (lowest SSIM in 3s window)
- ✅ Added context labeling (+/- 5 frames around transition)
- ✅ Batch processing for all 19 videos
- ✅ Ground truth timestamps loaded from data/ground_truth/
- ✅ Detailed logging for label verification

#### Labeling Algorithm:
| Step | Description | Implementation |
|------|-------------|----------------|
| 1. Load | Read manual timestamps from transitions.txt | ✅ Completed |
| 2. Snapper | Find lowest SSIM in +/- 1.5s window | ✅ Completed |
| 3. Label | Mark frame + 5 before/after as 1 | ✅ Completed |

#### Files Created:
| File | Purpose | Status |
|------|---------|--------|
| src/label_transitions.py | Automatic labeling engine | ✅ Created |
| label_all.ps1 | Batch labeling script | ✅ Created |
| logs/labeling.log | Labeling process logs | ✅ Auto-generated |

#### Machine Learning Ready:
**Dataset:** 19 videos with labeled transitions
- Positive class (label=1): Transition frames
- Negative class (label=0): Non-transition frames
- Ready for supervised learning!

### Version 1.1.0 - Enhanced Teacher Detection
**Date:** January 26, 2026  
**Status:** COMPLETED

#### Changes Made:
- ✅ Added HSV-based skin pixel detection for robust teacher tracking
- ✅ Created teacher_presence combined feature (black + skin pixels)
- ✅ Improved distinction between teacher motion vs slide transitions
- ✅ Created reprocess_videos.py utility script with auto-backup
- ✅ Updated documentation with new feature details
- ✅ Processed all 19 videos with enhanced feature set (25 features total)

#### New Features:
| Feature | Description | Purpose |
|---------|-------------|----------|
| skin_pixel_ratio | HSV-based skin detection [0,20,70] to [20,255,255] | Detect teacher's face/hands |
| teacher_presence | black_pixel_ratio + skin_pixel_ratio | Combined teacher detection metric |

#### Files Modified:
| File | Changes | Status |
|------|---------|--------|
| src/video_feature_extractor.py | Added skin detection & teacher_presence | ✅ Updated |
| src/reprocess_videos.py | New utility for re-processing with backup | ✅ Created |
| PROJECT_HISTORY_MASTER.md | Documentation updates | ✅ Updated |

#### Machine Learning Insight:
**Problem Solved:** The model can now distinguish:
- **Teacher Motion**: Low SSIM + High teacher_presence = NOT a transition
- **Slide Change**: Low SSIM + Low teacher_presence = TRANSITION

### Version 1.0.0 - Feature Extraction Module
**Date:** January 25, 2026  
**Status:** COMPLETED

#### Changes Made:
- ✅ Created comprehensive video feature extraction script
- ✅ Implemented all specialized features (Teacher Masking, SSIM, MSE, Edge Analysis)
- ✅ Implemented tri-zonal ROI analysis for bottom 20% of frame
- ✅ Added temporal sliding window with rolling statistics
- ✅ Implemented edge decay velocity calculation
- ✅ Added Min-Max normalization per video
- ✅ Created batch processing for multiple videos
- ✅ Added comprehensive logging and error handling
- ✅ Created requirements.txt with all dependencies
- ✅ Wrote detailed documentation (README.md)
- ✅ Updated master history file

#### Files Created:
| File | Purpose | Status |
|------|---------|--------|
| PROJECT_HISTORY_MASTER.md | Master documentation & history | ✅ Created |
| src/video_feature_extractor.py | Main feature extraction engine | ✅ Created |
| requirements.txt | Python package dependencies | ✅ Created |
| docs/README.md | Usage documentation | ✅ Created |
| data/videos/ | Input video directory | ✅ Created |
| data/output/ | Output CSV directory | ✅ Created |
| logs/ | Log file directory | ✅ Created |

#### Files Modified:
None

### Version 0.1.0 - Initial Setup
**Date:** January 25, 2026  
**Status:** COMPLETED

#### Changes Made:
- ✅ Created Project History Master File
- ✅ Set up project directory structure on D: drive

---

## Features & Modules

### Core Features
| Feature | Description | Status | Priority | Module |
|---------|-------------|--------|----------|--------|
| Video Feature Extraction | Extract 23+ features from videos | [x] | High | Core |
| Teacher Masking | Detect black pixels (teacher's t-shirt) | [x] | High | Core |
| Global Difference Analysis | SSIM, MSE, Histogram Correlation | [x] | High | Core |
| Edge Detection | Canny edge analysis with change rates | [x] | High | Core |
| Tri-Zonal ROI Analysis | Bottom 20% zone-specific features | [x] | High | Core |
| Temporal Sliding Window | 10-frame rolling statistics | [x] | High | Core |
| Edge Decay Velocity | Slope analysis for transition types | [x] | High | Core |
| Batch Video Processing | Process multiple videos automatically | [x] | High | Core |
| CSV Export | Normalized feature output per video | [x] | High | Core |
| Transition Detection (ML) | Train model on extracted features | [ ] | High | ML |
| Notes Generation | Auto-generate notes from transitions | [ ] | High | NLP |
| Note Organization | Categorize by topics/timestamps | [ ] | Medium | Organization |

### Modules Breakdown

#### Module 1: Video Feature Extraction (COMPLETED)
**Purpose:** Extract specialized features from lecture videos at 5 FPS for transition detection  
**Dependencies:** OpenCV, Scikit-Image, Pandas, NumPy  
**Status:** [x] Completed  
**File:** `src/video_feature_extractor.py`  
**Features Implemented:**
- Black pixel ratio (teacher masking)
- Global SSIM, MSE, histogram correlation
- Canny edge detection with change rates
- Tri-zonal ROI analysis (Left/Center/Right)
- Temporal sliding windows (10-frame/2-second)
- Edge decay velocity calculation
- Min-Max normalization per video
- Batch processing with logging

#### Module 2: Transition Classification (PLANNED)
**Purpose:** Train ML model to classify transitions using extracted features  
**Dependencies:** Scikit-learn, TensorFlow/PyTorch, XGBoost  
**Status:** [ ] Not Started  
**Planned Features:**
- Load and merge multiple CSV files
- Feature engineering and selection
- Train classification model (Random Forest, XGBoost, or Neural Network)
- Validate model performance
- Export trained model

#### Module 3: Note Generation (PLANNED)
**Purpose:** Generate structured notes from detected transitions  
**Dependencies:** OCR tools (Tesseract, EasyOCR), NLP libraries  
**Status:** [ ] Not Started  
**Planned Features:**
- Extract key frames at detected transitions
- OCR text extraction from slides
- Topic segmentation
- Generate markdown/PDF notes

---

## Technical Stack

### Core Processing
- **Language:** Python 3.8+
- **Video Processing:** OpenCV 4.8+
- **Image Analysis:** Scikit-Image 0.21+
- **Data Handling:** Pandas 2.0+, NumPy 1.24+
- **Scientific Computing:** SciPy 1.10+

### Machine Learning (Planned)
- **Framework:** Scikit-learn / XGBoost / PyTorch
- **Feature Engineering:** Custom + Scikit-learn
- **Model Types:** Random Forest, Gradient Boosting, Neural Networks

### OCR & NLP (Planned)
- **OCR:** Tesseract / EasyOCR / PaddleOCR
- **NLP:** NLTK / spaCy
- **Text Processing:** Regular expressions, pattern matching

### Frontend (Future)
- **Framework:** Streamlit / Gradio (for demo)
- **Web Framework:** Flask / FastAPI (for production)
- **UI Library:** Bootstrap / Tailwind CSS

### Database (Future)
- **Type:** SQLite (development) / PostgreSQL (production)
- **Engine:** SQLAlchemy ORM
- **Storage:** File-based CSV → Database migration planned

### DevOps & Deployment
- **Version Control:** Git (to be initialized)
- **Storage:** D:\ drive (local development)
- **Logging:** Python logging module → logs/
- **Documentation:** Markdown (this file + docs/README.md)
- **CI/CD:** [To be decided]
- **Deployment Platform:** [To be decided]

---

## Database Schema

### Tables Overview
[To be populated with database schema]

### Entity Relationship Diagram
[To be added]

---

## API Endpoints

### Authentication Endpoints
| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| POST | /api/auth/login | User login | [ ] |
| POST | /api/auth/register | User registration | [ ] |

### Notes Endpoints
| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| GET | /api/notes | Get all notes | [ ] |
| POST | /api/notes | Create note | [ ] |
| PUT | /api/notes/:id | Update note | [ ] |
| DELETE | /api/notes/:id | Delete note | [ ] |

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher installed
- pip package manager
- Video codec support (automatically handled by OpenCV)
- Minimum 4GB RAM, 8GB+ recommended
- 10GB+ free disk space

### Step-by-step Setup
1. **Navigate to project directory:**
   ```bash
   cd "D:\College_Life\projects\smart notes generator - trail 3"
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your videos:**
   - Copy lecture videos to `data/videos/` folder
   - Supported formats: MP4, AVI, MOV, MKV, FLV, WMV

4. **Run feature extraction:**
   ```bash
   # Process all videos in data/videos/
   python src/video_feature_extractor.py
   
   # Or process a single video
   python src/video_feature_extractor.py --single "data/videos/lecture.mp4"
   ```

5. **Check output:**
   - CSV files will be in `data/output/`
   - Logs will be in `logs/feature_extraction.log`

### Environment Variables
```bash
# No environment variables required for current version
# Future versions may use:
# - PYTHON_PATH (auto-detected)
# - OPENCV_VIDEOIO_DEBUG (for video debugging)
# - CUDA_VISIBLE_DEVICES (for GPU acceleration)
```

### Command-Line Options
```bash
python src/video_feature_extractor.py [OPTIONS]

Options:
  -i, --input PATH    Input folder (default: data/videos)
  -o, --output PATH   Output folder (default: data/output)
  -f, --fps INT       Target FPS (default: 5)
  -s, --single PATH   Process single video
  -h, --help          Show help
```

---

## Dependencies & Libraries

### Production Dependencies
| Package | Version | Purpose | Last Updated |
|---------|---------|---------|---------------|
| opencv-python | 4.8.0+ | Video processing & frame extraction | 2026-01-25 |
| opencv-contrib-python | 4.8.0+ | Additional OpenCV algorithms | 2026-01-25 |
| scikit-image | 0.21.0+ | SSIM, MSE, image metrics | 2026-01-25 |
| pandas | 2.0.0+ | Data manipulation & CSV export | 2026-01-25 |
| numpy | 1.24.0+ | Numerical computations | 2026-01-25 |
| scipy | 1.10.0+ | Scientific computing & statistics | 2026-01-25 |
| Pillow | 10.0.0+ | Image I/O operations | 2026-01-25 |
| pathlib2 | 2.3.7+ | Cross-platform path handling | 2026-01-25 |
| tqdm | 4.65.0+ | Progress bars (optional) | 2026-01-25 |

### Development Dependencies
| Package | Version | Purpose | Last Updated |
|---------|---------|---------|---------------|
| pytest | 7.4.0+ | Unit testing (planned) | Future |
| black | 23.0.0+ | Code formatting (planned) | Future |
| flake8 | 6.0.0+ | Linting (planned) | Future |
| jupyter | 1.0.0+ | Interactive analysis (planned) | Future |

### System Requirements
- **OS:** Windows 10/11 (Currently), Linux/macOS (Compatible)
- **RAM:** 4GB minimum, 8GB+ recommended
- **Storage:** 10GB+ free space (for videos and output)
- **Python Version:** 3.8 or higher ✅
- **GPU:** Optional (CUDA support for faster processing)
- **Video Codecs:** H.264/H.265 support recommended

### Installation Command
```bash
pip install -r requirements.txt
```

---

## Ideas & Future Enhancements

### Q1 2026 Ideas

- [ ] **Idea #1: ML Model Training for Transition Detection**
  - **Priority:** [x] High
  - **Complexity:** [x] Medium
  - **Estimated Time:** 15-20 hours
  - **Status:** Next Priority
  - **Notes:** Train Random Forest/XGBoost on extracted features to automatically classify transitions. Need to manually label ~500-1000 transition examples first.
  - **Dependencies:** Labeled training data from multiple videos

- [ ] **Idea #2: Real-time Video Processing**
  - **Priority:** [x] Medium
  - **Complexity:** [x] Complex
  - **Estimated Time:** 25-30 hours
  - **Status:** Under Review
  - **Notes:** Process live streams or real-time video feeds for instant transition detection. Requires optimization for speed.

- [ ] **Idea #3: Audio Analysis Integration**
  - **Priority:** [x] Medium
  - **Complexity:** [x] Medium
  - **Estimated Time:** 12-15 hours
  - **Status:** Under Review
  - **Notes:** Add audio features (silence detection, speech rate changes) to improve transition detection accuracy. Use librosa or pydub.

- [ ] **Idea #4: OCR Text Extraction from Slides**
  - **Priority:** [x] High
  - **Complexity:** [x] Medium
  - **Estimated Time:** 10-12 hours
  - **Status:** Planned
  - **Notes:** Extract text from detected slide transitions using Tesseract/EasyOCR. Essential for note generation.

### Q2 2026 Ideas

- [ ] **Idea #5: Automatic Summary Generation**
  - **Priority:** [x] High
  - **Complexity:** [x] Complex
  - **Estimated Time:** 20-25 hours
  - **Status:** Planned
  - **Notes:** Use NLP (GPT/BERT) to summarize extracted text into concise notes.

- [ ] **Idea #6: Multi-Language Support**
  - **Priority:** [ ] Medium
  - **Complexity:** [x] Medium
  - **Estimated Time:** 8-10 hours
  - **Status:** Future
  - **Notes:** Support Hindi, English mixed lectures common in PW videos.

- [ ] **Idea #7: Streamlit Web Interface**
  - **Priority:** [x] Medium
  - **Complexity:** [ ] Simple
  - **Estimated Time:** 6-8 hours
  - **Status:** Planned
  - **Notes:** Create user-friendly UI for uploading videos and viewing results.

### Q3 2026 Ideas

- [ ] **Idea #8: Batch Processing Optimization**
  - **Priority:** [ ] Low
  - **Complexity:** [x] Medium
  - **Estimated Time:** 10-12 hours
  - **Status:** Future
  - **Notes:** GPU acceleration with CUDA, parallel processing, frame caching.

- [ ] **Idea #9: Export to Notion/Obsidian**
  - **Priority:** [ ] Medium
  - **Complexity:** [ ] Simple
  - **Estimated Time:** 5-6 hours
  - **Status:** Future
  - **Notes:** Direct export to popular note-taking apps via API.

### Q4 2026 Ideas

- [ ] **Idea #10: Mobile App Version**
  - **Priority:** [ ] Low
  - **Complexity:** [x] Complex
  - **Estimated Time:** 40-50 hours
  - **Status:** Future
  - **Notes:** React Native or Flutter app for on-the-go processing.

- [ ] **Idea #11: Cloud Deployment**
  - **Priority:** [ ] Medium
  - **Complexity:** [x] Medium
  - **Estimated Time:** 15-20 hours
  - **Status:** Future
  - **Notes:** Deploy on AWS/GCP with queue-based video processing.

---

## Known Issues & Bugs

### Critical Issues
| Issue ID | Title | Description | Status | Assigned To | Due Date |
|----------|-------|-------------|--------|-------------|----------|
| BUG-001 | [Title] | [Description] | [ ] Open | [Name] | YYYY-MM-DD |

### High Priority Issues
| Issue ID | Title | Description | Status | Assigned To | Due Date |
|----------|-------|-------------|--------|-------------|----------|
| BUG-002 | [Title] | [Description] | [ ] Open | [Name] | YYYY-MM-DD |

### Medium/Low Priority Issues
| Issue ID | Title | Description | Status | Assigned To | Due Date |
|----------|-------|-------------|--------|-------------|----------|

---

## Performance Metrics

### Processing Performance
- **Frame Processing Speed:** ~5-10x faster than real-time (at 5 FPS sampling)
- **Video Processing Time:** ~6-12 minutes per hour of video content
- **Feature Extraction Rate:** ~300-500 frames per minute
- **CSV Generation Time:** < 1 second per video

### Resource Usage
- **Memory:** 500MB - 2GB (depends on video resolution)
  - 720p videos: ~500-800MB
  - 1080p videos: ~1-1.5GB
  - 4K videos: ~2-4GB
- **CPU:** Single-core processing (multi-threading planned)
  - Average utilization: 60-80% on single core
- **Storage:** 
  - CSV output: ~100-500KB per minute of video
  - Log files: ~10-50KB per video
  - Total project size: ~50MB (excluding videos)

### Accuracy Metrics (To Be Measured)
- **Feature Extraction Success Rate:** ~99% (handles corrupted frames gracefully)
- **Transition Detection Accuracy:** TBD (awaits ML model training)
- **False Positive Rate:** TBD
- **False Negative Rate:** TBD

### Scalability
- **Max Video Length:** Tested up to 3 hours (no theoretical limit)
- **Batch Processing:** Can process 10+ videos sequentially
- **Concurrent Processing:** Single-threaded (parallel processing planned)

### Uptime & Reliability
- **Error Handling:** Robust try-except blocks, graceful frame skip
- **Logging:** Comprehensive logging to logs/feature_extraction.log
- **Recovery:** Auto-skip corrupted frames, continue processing
- **Last Incident:** None (v1.0.0 stable)

---

## Development Workflow

### Git Workflow
- **Main Branch:** Production code
- **Develop Branch:** Development code
- **Feature Branches:** feature/feature-name

### Commit Message Format
```
[TYPE]: [Description]
- TYPE: feat (feature), fix (bug fix), docs (documentation), style, refactor, test, chore
```

### Code Review Process
1. Create feature branch
2. Make changes
3. Create Pull Request
4. Code review
5. Merge to develop
6. Test in staging
7. Deploy to production

---

## Documentation Standards

### Code Documentation
- Use JSDoc/JavaDoc/Docstring format
- Comment complex logic
- Include examples

### File Headers
```
/**
 * File: [filename]
 * Purpose: [What this file does]
 * Author: [Your name]
 * Created: YYYY-MM-DD
 * Last Modified: YYYY-MM-DD
 */
```

---

## Quick Reference

### Important Commands
```bash
# Process all videos with NEW features (skin detection + teacher presence)
python src/video_feature_extractor.py

# Re-process existing videos (auto-backup old CSVs)
python src/reprocess_videos.py

# Quick re-process (Windows)
reprocess_all.bat
# or PowerShell
.\reprocess_all.ps1

# Process single video
python src/video_feature_extractor.py --single "data/videos/lecture.mp4"

# Specify custom input/output folders
python src/video_feature_extractor.py --input "D:/my_videos" --output "D:/results"

# Change target FPS (default is 5)
python src/video_feature_extractor.py --fps 10

# Install all dependencies
pip install -r requirements.txt

# View help
python src/video_feature_extractor.py --help

# Check Python version
python --version

# View logs
cat logs/feature_extraction.log  # Linux/Mac
type logs\feature_extraction.log  # Windows
```

### Useful File Locations
- **Project Root:** `D:\College_Life\projects\smart notes generator - trail 3`
- **Input Videos:** `data/videos/`
- **Output CSVs:** `data/output/`
- **Logs:** `logs/feature_extraction.log`
- **Documentation:** `docs/README.md`
- **Master File:** `PROJECT_HISTORY_MASTER.md` (this file)

### Feature Columns in Output CSV
```
video_id, frame_index, timestamp_seconds,
black_pixel_ratio, skin_pixel_ratio, teacher_presence,
global_ssim, global_mse, histogram_correlation,
edge_count, edge_change_rate,
zone_left_ssim, zone_left_edge_density,
zone_center_ssim, zone_center_edge_density,
zone_right_ssim, zone_right_edge_density,
ssim_rolling_mean, ssim_rolling_std, ssim_rolling_max,
edge_rolling_mean, edge_rolling_std, edge_rolling_max,
edge_decay_velocity, label
```
**(25 total features - Updated in v1.1.0)**

### Key Feature Insights
- **Teacher Motion Detection**: High `teacher_presence` + Low `global_ssim` = Teacher moving (NOT transition)
- **Slide Transition**: Low `teacher_presence` + Low `global_ssim` = Slide change (TRANSITION)
- **Skin Detection**: HSV range [0,20,70] to [20,255,255] for robust face/hand detection

### Troubleshooting Quick Fixes
```bash
# Video won't open
# → Check codec compatibility (use VLC to verify)
# → Convert to H.264 MP4: ffmpeg -i input.avi -c:v libx264 output.mp4

# Out of memory error
# → Lower FPS: --fps 3
# → Process videos one at a time: --single

# Missing dependencies
# → Reinstall: pip install --upgrade -r requirements.txt

# Permission errors on logs/
# → Create logs folder: mkdir logs
```

---

## Contact & Collaboration

### Team Members
| Name | Role | Email | Phone |
|------|------|-------|-------|
| [Name] | Developer | [email] | [phone] |

### Resources
- **Repository:** D:\College_Life\projects\smart notes generator - trail 3
- **Documentation:** [Link to docs]
- **Project Management:** [Tool used]

---

## Update Instructions

**To update this file:**
1. Scroll to the relevant section
2. Update the information
3. Change the "Last Updated" date at the top
4. Save the file

**When to update:**
- After any code changes
- When adding new features
- When discovering bugs
- When completing tasks
- Weekly review recommended

---

**End of Document**

---

*This master file is the single source of truth for all project developments, modifications, and ideas. Keep it updated regularly for project clarity and team coordination.*







first prompt :

Write a professional Python script using OpenCV, Scikit-Image, and Pandas to extract features from lecture videos for transition detection.1. Core Logic:Process the video at 5 FPS.For every frame $t$, compare it to frame $t-1$.2. Specialized Feature Extraction:Teacher Masking: Identify pixels where RGB values are all < 50 (Teacher's black T-shirt). Calculate the 'Black Pixel Ratio'.Global Difference: Calculate $\Delta$ SSIM, Histogram Correlation, and Mean Squared Error (MSE) between consecutive frames.Edge Analysis: Use Canny Edge Detection. Calculate the total edge count and the 'Edge Change Rate' ($\Delta$ Edges).Tri-Zonal ROI: Divide the bottom 20% of the frame into 3 equal horizontal zones (Left, Center, Right). For each zone, calculate local $\Delta$ SSIM and local Edge Density.3. Temporal Memory (The Sliding Window):Create a 10-row (2-second) sliding window.For the Global SSIM and Edge Change, calculate: Rolling Mean, Rolling Std Dev, and Rolling Max.Calculate 'Edge Decay Velocity': The slope of edge count over the 10-frame window (to distinguish instant slide jumps from gradual manual erasing).4. Normalization & Output:Apply Min-Max Scaling to all features so they are between 0 and 1 (calculated per video).Output a CSV file with columns: video_id, frame_index, timestamp_seconds, all extracted features, and a label column initialized to 0.Handle the first 10 frames of the video by padding with the first valid calculation.
create csv file for each video, later i will merge them

23 features used:
video_id,frame_index,timestamp_seconds,black_pixel_ratio,global_ssim,global_mse,histogram_correlation,edge_count,edge_change_rate,zone_left_ssim,zone_left_edge_density,zone_center_ssim,zone_center_edge_density,zone_right_ssim,zone_right_edge_density,ssim_rolling_mean,ssim_rolling_std,ssim_rolling_max,edge_rolling_mean,edge_rolling_std,edge_rolling_max,edge_decay_velocity,label


Features : 
✅Uses refined predictions (ground truth labels from all 19 videos)
✅ 30 FPS extraction - Maximum quality frames
✅ Real-time teacher presence calculation on 300-frame windows
✅ Final slide capture - Adds transition at end of video
✅ Smart deduplication - SSIM > 0.93, keeps cleaner frame
✅ JSON metadata export - lecture_metadata.json with audio segments
✅ FFmpeg commands - Auto-generated for audio extraction

Output:
data/final_notes/<video>/ - High-quality PNG slides
data/final_notes/lecture_metadata.json - Metadata for AI
data/final_notes/extract_audio.bat - Windows audio extraction
data/final_notes/extract_audio.sh - Linux/Mac audio extraction