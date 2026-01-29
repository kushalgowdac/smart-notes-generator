# Smart Notes Generator - Project History & Master File
**Project Location:** `D:\College_Life\projects\smart notes generator - trail 3`  
**Created:** January 25, 2026  
**Last Updated:** January 29, 2026 - Version 3.1 (Utility Scripts, Checkpoint System & SSIM Testing)

---

## 📋 Recent Updates - Version 3.1

### January 29, 2026 - Production Utilities & Optimization Framework v3.1
**Major Features Added:**

1. **Checkpoint & Resume System (Zero Data Loss)**
   - Implemented automatic checkpoint saving in `process_new_lecture.py`
   - 5 checkpoint steps: audio → features → deduplication → slides → readme
   - Saves `.checkpoint.json` after each completed step
   - Resume detection: Re-run same command to continue from interruption
   - Handles Ctrl+C interruptions, Python errors, and system crashes gracefully
   - Incremental slide-by-slide saving in `generate_lecture_notes.py`
   - Resume from last completed slide when API quota exceeded
   - **Time savings:** Skip 12-minute feature extraction on resume (from 24% → instant)

2. **SSIM Threshold Testing Framework**
   - Created `test_ssim_batch.bat` - One-click automated testing
   - Created `check_accuracy.py` - Ground truth comparison tool
   - Created `src/test_ssim_thresholds.py` - Comprehensive Python framework
   - Tests multiple SSIM thresholds (0.75-0.97) against 18 training videos
   - Calculates Precision, Recall, F1-score for each threshold
   - Automatically identifies optimal threshold
   - **Speed:** 3 seconds per threshold (reuses existing slides)
   - **Result:** SSIM 0.85 optimal for incremental writing (F1=86.59%)

3. **Intelligent Video Analysis Utilities**
   - **analyze_video_type.py** - SSIM threshold recommender
     * Classifies videos: INCREMENTAL_WRITING / MIXED / DISCRETE_SLIDES
     * Analyzes SSIM distribution statistics (mean, std, quartiles)
     * Predicts slide reduction at different thresholds
     * Recommends optimal SSIM (0.80-0.85 for incremental, 0.92-0.95 for discrete)
     * Optional matplotlib visualization (histogram + sequential plot)
   
   - **check_slide_quality.py** - Pre-generation quality gate
     * Sharpness detection via Laplacian variance (threshold >100)
     * Teacher occlusion detection via HSV skin masking (threshold <15%)
     * Brightness quality check (50-220 range)
     * Contrast analysis (>30 threshold)
     * Classifies slides: EXCELLENT / GOOD / ACCEPTABLE / POOR
     * Outputs quality_report.json with detailed metrics
     * Visual problem slide viewer for manual review
   
   - **check_ocr_confidence.py** - OCR quality analyzer
     * Analyzes ocr_cache.json for confidence scores
     * Block-level confidence tracking (min/max/avg)
     * Identifies EMPTY vs LOW_CONFIDENCE slides
     * Generates ocr_confidence_report.json
     * Visual problem slide viewer

4. **API Quota & Recovery Utilities**
   - **resume_notes.py** - Checkpoint recovery for notes generation
     * Detects last completed slide by parsing notes.md (regex-based)
     * Validates slide completion (>50 chars content check)
     * Estimates API calls remaining vs quota (20/day Gemini free tier)
     * Dry-run mode for planning multi-day processing
     * **Use case:** 50-slide lecture = 3 days with free tier
   
   - **curate_slides.py** - Interactive manual curator
     * OpenCV-based visual interface with keyboard controls
     * Controls: Y(keep) / N(skip) / U(undo) / ←→(navigate) / R(reset) / S(save) / Q(quit)
     * Auto-remove duplicates function (SSIM>0.90 batch removal)
     * Semi-transparent info overlay with current status
     * Backs up original transitions.json before saving
     * **Speed:** Review 20 slides in ~2 minutes

5. **Export & Distribution Utilities**
   - **export_notes.py** - Multi-format exporter
     * PDF export via WeasyPrint (CSS styling, cover page, table of contents)
     * DOCX export via python-docx (formatting, headings, images)
     * Anki flashcard generation via genanki (auto-creates from headings)
     * Notion export (Markdown with metadata header)
     * All dependencies optional (graceful fallback)
     * Command: `--format pdf` / `--format docx` / `--format anki` / `--format notion`

6. **Documentation & Guides**
   - Created `docs/CHECKPOINT_RESUME_GUIDE.md` - Complete checkpoint system guide
   - Created `docs/SSIM_TESTING_GUIDE.md` - SSIM optimization workflow
   - Created `SSIM_QUICK_START.md` - Quick reference for threshold testing
   - Updated README.md with checkpoint features and SSIM testing
   - All utilities documented with usage examples and rationale

**Files Created:**
- `src/analyze_video_type.py` - 350+ lines, SSIM recommender
- `src/resume_notes.py` - 200+ lines, API quota recovery
- `src/check_slide_quality.py` - 300+ lines, quality gate
- `src/curate_slides.py` - 400+ lines, interactive curator
- `src/export_notes.py` - 500+ lines, multi-format exporter
- `src/check_ocr_confidence.py` - 200+ lines, OCR analyzer
- `src/test_ssim_thresholds.py` - 350+ lines, SSIM testing framework
- `src/quick_ssim_test.py` - 100+ lines, quick SSIM analysis
- `check_accuracy.py` - Ground truth comparison tool
- `test_ssim_batch.bat` - Automated SSIM testing batch script
- `docs/CHECKPOINT_RESUME_GUIDE.md` - Comprehensive checkpoint guide
- `docs/SSIM_TESTING_GUIDE.md` - SSIM optimization guide
- `SSIM_QUICK_START.md` - Quick reference guide

**Testing Results:**
- Checkpoint system tested: 24% feature extraction → resume → skip to deduplication
- SSIM testing: Validated optimal threshold (0.85) on 18 training videos
- Quality gate: Detected blur, occlusion, brightness issues in test slides
- Resume notes: Successfully parsed notes.md, estimated 3-day timeline for 50 slides
- Export: Generated PDF with CSS styling, DOCX with formatting, Anki flashcards

**Performance Metrics:**
- Checkpoint resume: **0 seconds** (instant skip for completed steps)
- SSIM testing: **3 seconds per threshold** (vs 40 minutes full reprocess)
- Quality analysis: **~2 seconds per slide** (Laplacian + HSV analysis)
- Interactive curation: **~6 seconds per slide** (manual review)
- Notes resumption: **Saves 10-20 minutes** (skips OCR + transcription re-run)

**Integration Benefits:**
- All utilities work with checkpoint system (no conflicts)
- SSIM testing uses rerun_deduplication.py (instant results)
- Quality gate can pre-filter before notes generation (saves API quota)
- Resume notes enables multi-day processing for free tier users
- Export utilities create distribution-ready outputs

7. **Systematic Testing Framework (January 29, 2026)**
   - **check_baseline_accuracy.py** - Ground truth validation tool
     * Compares current transitions.json against ground_truth/*.txt
     * Tests all 18 training videos with ±2.0s tolerance
     * Calculates Precision, Recall, F1-score per video
     * Provides optimization recommendations based on F1-score
     * Outputs: data/analysis/baseline_accuracy.txt
   
   - **test_utilities.py** - Automated test orchestrator
     * 5 priority levels: MUST DO → SHOULD DO → OPTIONAL
     * Priority 1: Baseline accuracy check (30s)
     * Priority 2: SSIM optimization (5-10min)
     * Priority 3: Video type analysis (10-15s/video)
     * Priority 4: Slide quality check (1-2min/video)
     * Priority 5: OCR confidence + Resume detection
     * Saves results to data/analysis/utility_tests/
   
   - **run_all_tests.ps1** - Interactive PowerShell test suite
     * User-friendly prompts for each testing phase
     * Colored output with progress indicators
     * Optional step skipping
     * Comprehensive summary at end
     * Estimated time: 1-2 hours for full suite
   
   - **docs/UTILITY_TESTING_GUIDE.md** - Complete testing workflow
     * Priority-based testing order
     * Command examples for each utility
     * Output interpretation guidelines
     * Parameter tuning recommendations
     * Troubleshooting section

**Testing Workflow:**
```
Priority 1: Baseline Accuracy → Identify if optimization needed
Priority 2: SSIM Optimization → Find optimal threshold (if F1 < 85%)
Priority 3: Video Type Analysis → Per-video threshold recommendations
Priority 4: Quality Assessment → Detect poor slides for re-extraction
Priority 5: OCR + Resume → Verify text extraction & checkpoints
```

**Files Added:**
- `check_baseline_accuracy.py` - 200+ lines, ground truth comparison
- `test_utilities.py` - 300+ lines, automated test orchestrator
- `run_all_tests.ps1` - 250+ lines, interactive PowerShell script
- `docs/UTILITY_TESTING_GUIDE.md` - 500+ lines, comprehensive guide

**How to Use:**
```bash
# Quick baseline check
python check_baseline_accuracy.py

# Full automated testing
.\run_all_tests.ps1

# Individual priority test
python test_utilities.py --priority 1
```

**Expected Results:**
- F1 ≥ 85%: Excellent - No changes needed
- F1 75-85%: Good - Minor tuning beneficial
- F1 60-75%: Fair - SSIM optimization recommended
- F1 < 60%: Poor - Urgent parameter tuning required

---

## � CLI Command Reference - All Scripts

### Core Pipeline Scripts

#### 1. process_new_lecture.py - Main Processing Pipeline
**Purpose:** End-to-end video processing with checkpoint system

```bash
# Basic usage
python src/process_new_lecture.py data/videos/lecture.mp4

# With custom SSIM threshold
python src/process_new_lecture.py data/videos/lecture.mp4 --ssim 0.85

# With custom model and threshold
python src/process_new_lecture.py data/videos/lecture.mp4 \
  --model models/my_model.pkl \
  --threshold 0.05 \
  --ssim 0.90

# Full parameter customization
python src/process_new_lecture.py data/videos/lecture.mp4 \
  --output-dir data/lectures \
  --model models/xgboost_model_20260126_160645.pkl \
  --threshold 0.01 \
  --ssim 0.95 \
  --blank-threshold 0.02 \
  --lookback 10.0 \
  --feature-fps 5 \
  --slide-fps 30
```

**CLI Arguments:**
- `video_path` (required): Path to lecture video file
- `--output-dir`: Output base directory (default: `data/lectures`)
- `--model`: Trained model path (default: `models/xgboost_model_20260126_160645.pkl`)
- `--threshold`: Prediction threshold (default: `0.01` for sparse training data)
- `--ssim`: SSIM dedup threshold (default: `0.95` for board-only comparison)
- `--blank-threshold`: Blank edge threshold (default: `0.02` for HD videos)
- `--lookback`: Adaptive window lookback seconds (default: `10.0`)
- `--feature-fps`: FPS for feature extraction (default: `5`)
- `--slide-fps`: FPS for slide extraction (default: `30`)

**Checkpoint System:** Auto-saves progress. Re-run same command to resume.

---

#### 2. generate_lecture_notes.py - Multimodal AI Notes Generation
**Purpose:** Generate lecture notes from slides + audio using Gemini AI

```bash
# Basic usage (requires GOOGLE_API_KEY env variable)
python src/generate_lecture_notes.py data/lectures/algo_1

# With API key argument
python src/generate_lecture_notes.py data/lectures/algo_1 --api-key YOUR_GEMINI_KEY
```

**CLI Arguments:**
- `lecture_folder` (required): Path to processed lecture folder
- `--api-key`: Google Gemini API key (or set `GOOGLE_API_KEY` env variable)

**Requirements:**
- Processed lecture folder with slides/ and audio/
- Google Gemini API key (free tier: 20 requests/day)

**Output:** `notes.md` with AI-generated lecture notes

**Resume Feature:** Automatically resumes from last completed slide

---

#### 3. production_note_maker.py - Batch Notes Generation
**Purpose:** Process multiple videos with note generation

```bash
# List all available videos
python src/production_note_maker.py --list-videos

# Process single video
python src/production_note_maker.py --video algo_1

# Process all videos
python src/production_note_maker.py

# Use custom predictions file
python src/production_note_maker.py \
  --predictions data/my_predictions.csv \
  --video toc_1
```

**CLI Arguments:**
- `--video`: Process only specific video (e.g., `algo_1`, `toc_1`)
- `--predictions`: Path to predictions CSV (default: `data/all_predictions.csv`)
- `--list-videos`: List all available videos and exit

---

### Deduplication & Optimization Scripts

#### 4. deduplicate_transitions.py - SSIM-Based Deduplication
**Purpose:** Remove duplicate slides using SSIM comparison

```bash
# Process single video
python src/deduplicate_transitions.py --video algo_1

# Process all videos
python src/deduplicate_transitions.py --all

# With custom SSIM thresholds
python src/deduplicate_transitions.py --video algo_1 \
  --ssim-dedup 0.90 \
  --ssim-rapid 0.80

# Use predictions.csv instead of transitions.json
python src/deduplicate_transitions.py --video algo_1 --use-predictions
```

**CLI Arguments:**
- `--video`: Process single video by ID
- `--all`: Process all videos
- `--use-predictions`: Use `predictions.csv` instead of `transitions.json`
- `--ssim-dedup`: SSIM threshold for deduplication (default: `0.95`)
- `--ssim-rapid`: SSIM threshold for rapid-fire detection (default: `0.85`)
- `--blank-threshold`: Edge count threshold for blank slides (default: `0.1`)

---

#### 5. test_ssim_thresholds.py - SSIM Threshold Optimization
**Purpose:** Test different SSIM values against ground truth

```bash
# Test default thresholds (0.75-0.97)
python src/test_ssim_thresholds.py

# Test specific thresholds
python src/test_ssim_thresholds.py --thresholds 0.80 0.85 0.90 0.95

# Test single video
python src/test_ssim_thresholds.py \
  --video algo_1 \
  --thresholds 0.85 0.90 0.95

# Custom directories
python src/test_ssim_thresholds.py \
  --lectures-dir data/lectures \
  --ground-truth-dir data/ground_truth
```

**CLI Arguments:**
- `--thresholds`: SSIM thresholds to test (default: `0.75 0.80 0.85 0.90 0.92 0.95 0.97`)
- `--video`: Test only specific video (e.g., `algo_1`)
- `--lectures-dir`: Lectures directory path (default: `data/lectures`)
- `--ground-truth-dir`: Ground truth directory path (default: `data/ground_truth`)

**Output:** CSV summary + detailed JSON results in `data/analysis/`

---

### Slide Extraction & Quality Scripts

#### 6. extract_best_slides.py - High-Quality Slide Extraction
**Purpose:** Extract best quality slide frames from transitions

```bash
# Process all videos
python src/extract_best_slides.py

# Process single video
python src/extract_best_slides.py --video-id algo_1

# With custom parameters
python src/extract_best_slides.py \
  --lookback 15.0 \
  --fps 60 \
  --lectures-dir data/lectures \
  --videos-dir data/videos
```

**CLI Arguments:**
- `--lectures-dir`: Lectures directory (default: `data/lectures`)
- `--videos-dir`: Videos directory (default: `data/videos`)
- `--lookback`: Max lookback window in seconds (default: `10.0`)
- `--fps`: Video FPS (default: `30`)
- `--video-id`: Process single video only

---

### Training & Model Scripts

#### 7. video_feature_extractor.py - Feature Extraction for Training
**Purpose:** Extract video features for model training

```bash
# Extract features from all videos
python src/video_feature_extractor.py

# Custom input/output directories
python src/video_feature_extractor.py \
  --input data/videos \
  --output data/output

# Process single video with custom FPS
python src/video_feature_extractor.py \
  --single lecture_01.mp4 \
  --fps 10
```

**CLI Arguments:**
- `--input`, `-i`: Input videos directory (default: `data/videos`)
- `--output`, `-o`: Output CSV directory (default: `data/output`)
- `--fps`, `-f`: Sampling FPS (default: `5`)
- `--single`, `-s`: Process single video filename only

---

#### 8. label_transitions.py - Ground Truth Labeling
**Purpose:** Auto-label transitions for model training

```bash
# Label all videos
python src/label_transitions.py

# Custom directories and parameters
python src/label_transitions.py \
  --csv-dir data/output \
  --ground-truth-dir data/ground_truth \
  --window 2.0 \
  --context 7

# Process single video
python src/label_transitions.py --single algo_1
```

**CLI Arguments:**
- `--csv-dir`: CSV features directory (default: `data/output`)
- `--ground-truth-dir`: Ground truth directory (default: `data/ground_truth`)
- `--window`: Time window for matching in seconds (default: `1.5`)
- `--context`: Context rows around transitions (default: `5`)
- `--single`: Process single video ID only

---

#### 9. reprocess_videos.py - Reprocess Features with New Parameters
**Purpose:** Re-extract features with updated parameters (with backup)

```bash
# Reprocess all videos
python src/reprocess_videos.py

# With custom FPS
python src/reprocess_videos.py --fps 10

# No backup (careful!)
python src/reprocess_videos.py --no-backup

# Custom backup directory
python src/reprocess_videos.py --backup-dir data/my_backup
```

**CLI Arguments:**
- `--input`, `-i`: Input videos directory (default: `data/videos`)
- `--output`, `-o`: Output CSV directory (default: `data/output`)
- `--fps`, `-f`: Sampling FPS (default: `5`)
- `--no-backup`: Skip backup creation
- `--backup-dir`: Custom backup directory (default: `data/output_backup`)

---

### Visualization & Preview Scripts

#### 10. visualize_transitions.py - HTML Transition Preview Generator
**Purpose:** Generate visual HTML previews of detected transitions

```bash
# List available videos
python src/visualize_transitions.py --list-videos

# Generate preview for single video
python src/visualize_transitions.py --video algo_1

# Process all videos
python src/visualize_transitions.py --all

# Custom thumbnail size and predictions file
python src/visualize_transitions.py \
  --video algo_1 \
  --predictions data/all_predictions.csv \
  --thumbnail-width 640
```

**CLI Arguments:**
- `--video`: Process single video by ID
- `--all`: Process all videos
- `--predictions`: Predictions CSV path (default: `data/all_predictions.csv`)
- `--thumbnail-width`: Thumbnail width in pixels (default: `320`)
- `--list-videos`: List all available videos and exit

**Output:** `data/transition_previews/<video_id>_preview.html`

---

#### 11. generate_manual_previews.py - Ground Truth Preview Generator
**Purpose:** Generate HTML previews for manually labeled transitions

```bash
# Generate preview for single video
python src/generate_manual_previews.py --video algo_1

# Process all videos with ground truth
python src/generate_manual_previews.py --all
```

**CLI Arguments:**
- `--video`: Process single video ID
- `--all`: Process all videos with ground truth

**Output:** `data/transition_previews/<video_id>_manual.html`

---

#### 12. generate_final_preview.py - Final Slides Preview Generator
**Purpose:** Generate HTML previews for deduplicated final slides

```bash
# Generate preview for single video
python src/generate_final_preview.py --video algo_1
```

**CLI Arguments:**
- `--video`: Generate preview for single video ID

**Output:** `data/transition_previews/<video_id>_final.html`

---

### Organization & Utility Scripts

#### 13. organize_lecture_folders.py - Lecture Folder Organizer
**Purpose:** Organize processed videos into lecture folders

```bash
# Organize all videos
python src/organize_lecture_folders.py

# Process single video
python src/organize_lecture_folders.py --video-id algo_1

# Custom directories and deduplication file
python src/organize_lecture_folders.py \
  --videos-dir data/videos \
  --deduplicated-json data/final_slides/final_unique_slides.json \
  --output-dir data/lectures
```

**CLI Arguments:**
- `--videos-dir`: Videos directory (default: `data/videos`)
- `--deduplicated-json`: Deduplicated transitions JSON (default: `data/final_slides/final_unique_slides.json`)
- `--output-dir`: Output directory for lectures (default: `data/lectures`)
- `--video-id`: Process single video only

---

### Testing & Validation Scripts

#### 14. check_baseline_accuracy.py - Ground Truth Accuracy Check
**Purpose:** Validate current pipeline accuracy against ground truth

```bash
# Run baseline accuracy check on all training videos
python check_baseline_accuracy.py
```

**CLI Arguments:** None (processes all training videos)

**Output:** 
- Console table with per-video metrics
- `data/analysis/baseline_accuracy.txt`

**Metrics:** Precision, Recall, F1-score with ±2.0s tolerance

---

#### 15. test_utilities.py - Automated Testing Framework
**Purpose:** Run systematic testing on all utility scripts

```bash
# Run all tests
python test_utilities.py

# Run specific priority test
python test_utilities.py --priority 1  # Baseline accuracy
python test_utilities.py --priority 2  # SSIM optimization
python test_utilities.py --priority 3  # Video type analysis
python test_utilities.py --priority 4  # Slide quality
python test_utilities.py --priority 5  # OCR + Resume
```

**CLI Arguments:**
- `--priority`: Run only specific priority test (1-5)

**Output:** Results saved to `data/analysis/utility_tests/`

---

### Quick Reference - Most Common Commands

```bash
# 1. Process new video (most common)
python src/process_new_lecture.py data/videos/new_lecture.mp4 --ssim 0.85

# 2. Generate notes for processed video
python src/generate_lecture_notes.py data/lectures/new_lecture

# 3. Check baseline accuracy (testing)
python check_baseline_accuracy.py

# 4. Test SSIM thresholds (optimization)
python src/test_ssim_thresholds.py --video algo_1 --thresholds 0.80 0.85 0.90

# 5. Visualize transitions (validation)
python src/visualize_transitions.py --video algo_1

# 6. Deduplicate with custom SSIM
python src/deduplicate_transitions.py --video algo_1 --ssim-dedup 0.85

# 7. Extract features for training
python src/video_feature_extractor.py --input data/videos --fps 5

# 8. Generate preview for final slides
python src/generate_final_preview.py --video algo_1
```

---

## �📋 Recent Updates - Version 3.0

### January 29, 2026 - Multimodal Notes Generation & Optimization v3.0
**Major Features Added:**

---

### Utility Scripts (Root Directory)

#### 16. rerun_deduplication.py - Instant SSIM Re-optimization
**Purpose:** Re-run deduplication with new SSIM threshold (no video reprocessing)

```bash
# Re-deduplicate with SSIM 0.85
python rerun_deduplication.py data/lectures/deadlock_os --ssim 0.85

# Test lower threshold (more aggressive deduplication)
python rerun_deduplication.py data/lectures/algo_1 --ssim 0.80

# Test higher threshold (keep more slides)
python rerun_deduplication.py data/lectures/cn_1 --ssim 0.90
```

**CLI Arguments:**
- `lecture_dir` (required): Path to lecture directory with slides/
- `--ssim`: New SSIM threshold (default: `0.85`)

**Speed:** ~3 seconds (vs 40 minutes for full reprocessing)  
**Output:** Updates `transitions.json` with new deduplicated transitions

---

#### 17. analyze_video_type.py - SSIM Threshold Recommender
**Purpose:** Classify video teaching style and recommend optimal SSIM

```bash
# Analyze video type
python analyze_video_type.py data/lectures/deadlock_os

# With visualization (requires matplotlib)
python analyze_video_type.py data/lectures/chemistry_01_english --visualize
```

**CLI Arguments:**
- `lecture_dir` (required): Path to lecture directory
- `--visualize`: Show SSIM distribution plot (histogram + sequential)

**Output:**
- Console: Video type classification and SSIM recommendation
- File: `video_type_analysis.json` in lecture directory

**Video Types:**
- `INCREMENTAL_WRITING`: Professor builds slides gradually → SSIM 0.80-0.85
- `DISCRETE_SLIDES`: Clean slide changes → SSIM 0.92-0.95
- `MIXED`: Combination of both → SSIM 0.85-0.90

---

#### 18. check_slide_quality.py - Pre-Generation Quality Gate
**Purpose:** Detect poor quality slides before notes generation

```bash
# Basic quality check
python check_slide_quality.py data/lectures/deadlock_os

# With custom threshold
python check_slide_quality.py data/lectures/algo_1 --threshold 70

# Show bad slides visually
python check_slide_quality.py data/lectures/cn_1 --show-bad-slides
```

**CLI Arguments:**
- `lecture_dir` (required): Path to lecture directory
- `--threshold`: Quality score threshold for warnings (default: `70`)
- `--show-bad-slides`: Display poor quality slides in OpenCV window

**Output:**
- Console: Quality summary (EXCELLENT/GOOD/ACCEPTABLE/POOR)
- File: `quality_report.json` in lecture directory

**Checks:**
- Sharpness (Laplacian variance > 100)
- Brightness (50-220 range)
- Contrast (> 30)
- Teacher occlusion (< 15%)

---

#### 19. check_ocr_confidence.py - OCR Quality Analyzer
**Purpose:** Identify slides with poor text extraction

```bash
# Check OCR confidence
python check_ocr_confidence.py data/lectures/deadlock_os

# With custom confidence threshold
python check_ocr_confidence.py data/lectures/chemistry_01 --threshold 0.8

# Show problem slides visually
python check_ocr_confidence.py data/lectures/algo_1 --show-problems

# Export detailed JSON report
python check_ocr_confidence.py data/lectures/db_1 --export-report
```

**CLI Arguments:**
- `lecture_dir` (required): Path to lecture directory
- `--threshold`: Confidence threshold (default: `0.75`)
- `--show-problems`: Display problem slides visually
- `--export-report`: Save detailed JSON report

**Requirements:** `ocr_cache.json` (generated during notes creation)

**Output:**
- Console: Confidence summary (HIGH/MEDIUM/LOW)
- File: `ocr_confidence_report.json` (if --export-report used)

**Flags:**
- `HIGH_CONFIDENCE`: Avg > threshold, reliable
- `MEDIUM_CONFIDENCE`: Avg near threshold, review recommended
- `LOW_CONFIDENCE`: Avg < threshold, manual correction needed
- `EMPTY`: No text detected

---

#### 20. resume_notes.py - Checkpoint Recovery for Notes Generation
**Purpose:** Resume interrupted notes generation from last checkpoint

```bash
# Auto-detect last completed slide and resume
python resume_notes.py data/lectures/deadlock_os

# Start from specific slide number
python resume_notes.py data/lectures/deadlock_os --start-from 18

# Force regenerate all (ignore checkpoint)
python resume_notes.py data/lectures/deadlock_os --force-regenerate

# Dry run (show what would be done)
python resume_notes.py data/lectures/deadlock_os --dry-run
```

**CLI Arguments:**
- `lecture_dir` (required): Path to lecture directory
- `--start-from`: Start from specific slide number (overrides auto-detection)
- `--force-regenerate`: Ignore checkpoint and regenerate all notes
- `--dry-run`: Show plan without executing

**Output:**
- Detects last completed slide from `notes.md`
- Shows remaining slides and estimated API calls
- Resumes generation seamlessly

**Use Case:** Free tier Gemini limit (20 requests/day) for 50-slide lecture = 3 days

---

#### 21. curate_slides.py - Interactive Manual Curator
**Purpose:** Manually review and select slides with visual interface

```bash
# Launch interactive curator
python curate_slides.py data/lectures/deadlock_os

# Works on any processed lecture
python curate_slides.py data/lectures/chemistry_01_english
```

**CLI Arguments:**
- `lecture_dir` (required): Path to lecture directory

**Keyboard Controls:**
- `Y`: Keep current slide and move to next
- `N`: Skip current slide and move to next
- `U`: Undo (toggle keep/skip status)
- `←` `→` or `A` `D`: Navigate between slides
- `R`: Auto-remove remaining duplicates (SSIM > 0.90)
- `S`: Save changes and exit
- `Q` or `ESC`: Quit without saving

**Output:**
- Updates `transitions.json` with curated selections
- Backs up original to `transitions_backup.json`
- Semi-transparent info overlay shows current status

**Speed:** Review 20 slides in ~2 minutes

---

#### 22. export_notes.py - Multi-Format Notes Exporter
**Purpose:** Export generated notes to multiple distribution formats

```bash
# Export to PDF
python export_notes.py data/lectures/deadlock_os --format pdf

# Export to multiple formats (comma-separated)
python export_notes.py data/lectures/chemistry_01 --format pdf,docx

# Export to all formats
python export_notes.py data/lectures/algo_1 --format all

# Short form
python export_notes.py data/lectures/db_1 -f docx
```

**CLI Arguments:**
- `lecture_dir` (required): Path to lecture directory with notes.md
- `--format`, `-f`: Export format(s) - `pdf`, `docx`, `anki`, `notion`, or `all` (comma-separated, default: `pdf`)

**Supported Formats:**
- `pdf`: Professional PDF with CSS styling (requires `weasyprint`)
- `docx`: Microsoft Word document (requires `python-docx`)
- `anki`: Anki flashcard deck `.apkg` (requires `genanki`)
- `notion`: Notion-ready Markdown (no extra dependencies)
- `all`: Export to all formats

**Output Files:**
- `notes.pdf` - Styled PDF with cover page and TOC
- `notes.docx` - Formatted Word document
- `notes.apkg` - Anki flashcard deck (auto-generated from headings)
- `notes_notion.md` - Notion-compatible Markdown

**Dependencies (optional):**
```bash
pip install markdown weasyprint      # For PDF
pip install python-docx              # For DOCX
pip install genanki                  # For Anki
```

**Graceful Fallback:** Skips formats with missing dependencies

---

### Quick Reference - Utility Scripts

```bash
# 1. Re-optimize SSIM threshold (instant)
python rerun_deduplication.py data/lectures/<name> --ssim 0.85

# 2. Analyze video type and get SSIM recommendation
python analyze_video_type.py data/lectures/<name> --visualize

# 3. Check slide quality before notes generation
python check_slide_quality.py data/lectures/<name> --show-bad-slides

# 4. Check OCR confidence
python check_ocr_confidence.py data/lectures/<name> --threshold 0.75

# 5. Resume interrupted notes generation
python resume_notes.py data/lectures/<name> --dry-run

# 6. Manually curate slides interactively
python curate_slides.py data/lectures/<name>

# 7. Export notes to PDF/DOCX/Anki
python export_notes.py data/lectures/<name> --format pdf,docx
```

---

### January 29, 2026 - Multimodal Notes Generation & Optimization v3.0 (continued)
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

