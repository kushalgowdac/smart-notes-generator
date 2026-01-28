# 🎓 Smart Notes Generator

An intelligent system that automatically processes lecture videos to extract key slides and generate comprehensive study notes using computer vision and multimodal AI.

## ✨ Features

- **Automated Slide Detection**: XGBoost ML model with 98.68% accuracy
- **Smart Deduplication**: SSIM-based duplicate removal with configurable thresholds
- **Multimodal Notes Generation**: Combines OCR + Audio Transcription + AI synthesis
- **Teacher Occlusion Prevention**: Automatically selects frames without teacher blocking content
- **Instant Re-optimization**: Re-run deduplication without reprocessing video (3s vs 40min)
- **Production Ready**: Tested on 19+ real lecture videos (Physics Wallah, etc.)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg (for audio extraction)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/smart-notes-generator.git
cd smart-notes-generator
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API keys** (for notes generation)
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Usage

## 🎯 Core Scripts (Start Here)

### 1. Process New Lecture Video (Main Pipeline)
```bash
# Full pipeline: slides extraction + notes generation
python src/process_new_lecture.py data/videos/my_lecture.mp4

# Slides only (skip notes generation to save API quota)
python src/process_new_lecture.py data/videos/my_lecture.mp4 --skip-notes
```

**What it does:**
- Extracts features at 5 FPS
- Predicts transitions using XGBoost model
- Deduplicates with SSIM
- Extracts best quality frames at 30 FPS
- Organizes output in `data/lectures/<video_id>/`

---

### 2. Generate Notes from Existing Slides
```bash
python src/production_note_maker.py data/lectures/my_lecture
```

**What it does:**
- OCR text extraction from slides (PaddleOCR)
- Audio transcription (Whisper)
- AI note generation (Gemini 2.5 Flash)
- Smart caching for instant re-runs
- Outputs: `notes.md` with structured content

**API Quota:** Free tier = 20 requests/day

---

### 3. Convenience Wrapper (Optional)
```bash
# Windows
.\process_complete.bat data\videos\my_lecture.mp4

# Linux/Mac
./process_complete.sh data/videos/my_lecture.mp4
```

Runs both slides extraction + notes generation automatically.

---

## 🔧 Utility Scripts (Advanced Use)

### Re-optimize Deduplication (Zero Video Processing)
```bash
# Adjust SSIM threshold without reprocessing video (3s vs 40min!)
python rerun_deduplication.py data/lectures/my_lecture --ssim 0.85

# Try different thresholds
python rerun_deduplication.py data/lectures/my_lecture --ssim 0.80  # Stricter
python rerun_deduplication.py data/lectures/my_lecture --ssim 0.92  # Moderate
```

**When to use:**
- Too many duplicate slides detected
- Incremental board writing videos (use 0.80-0.85)
- Want to optimize without waiting 40 minutes

**Output:** `transitions_reprocessed_ssim0.85.json` (doesn't overwrite original)

---

### Generate HTML Preview (Verify Slides Before Notes)
```bash
python src/generate_final_preview.py data/lectures/my_lecture
```

**What it does:**
- Creates HTML gallery with all detected slides
- Shows timestamps and SSIM scores
- Helps verify quality before spending API quota on notes
- Output: `data/final_slides/<video>_final_slides.html`

---

### Quick Notes Generation (Shortcuts)
```bash
# Windows
.\generate_notes.bat my_lecture

# Linux/Mac
./generate_notes.sh my_lecture
```

Shortcut for `python src/production_note_maker.py data/lectures/my_lecture`

---

### Test OCR Installation
```bash
python test_paddle.py
```

**When to use:**
- After installing dependencies
- Troubleshooting PaddleOCR issues
- Verifying CPU/GPU detection

---

### Batch Process Multiple Videos
```bash
# Windows
.\reprocess_all.bat

# PowerShell (with progress tracking)
.\reprocess_all.ps1
```

Processes all videos in `data/videos/` automatically.

---

### Manual Labeling (Create Training Data)
```bash
python src/label_transitions.py
```

**When to use:**
- Creating ground truth for model training
- Have manual transition timestamps in `data/ground_truth/<video>/transitions.txt`
- Want to improve model accuracy on your specific video type

**Input format (transitions.txt):**
```
84.4
405.39
504.39
```

---

## 🧪 Analysis Scripts (For Development)

### Compare Deduplication Approaches
```bash
python compare_deduplication.py
```
Compares clustering-only vs SSIM deduplication across all videos.

### Extract with Clustering Only
```bash
python extract_with_clustering_only.py
```
Tests clustering layer without SSIM (for algorithm analysis).

### Visualize Transitions
```bash
python src/visualize_transitions.py data/lectures/my_lecture
```
Generates HTML preview of detected transitions.

---

## 📚 Training Scripts (For Model Development)

### Train New XGBoost Model
```bash
python src/train_xgboost_model.py
```
Requires: `data/master_dataset.csv` with labeled transitions.

### Merge & Split Dataset
```bash
python src/merge_and_split_dataset.py
```
Combines individual video CSVs into train/test split.

---

## 🗂️ Data Organization Scripts

### Organize Lecture Folders
```bash
python src/organize_lecture_folders.py
```
Restructures old outputs into new folder format.

### Reprocess All Videos
```bash
# Windows
.\reprocess_all.bat

# PowerShell
.\reprocess_all.ps1
```
Batch processing for all videos in `data/videos/`.

---

## ⚙️ Installation Helpers

### Install Notes Dependencies Only
```bash
# If you only need notes generation (not CV pipeline)
.\install_notes_deps.ps1
```

Installs: PaddleOCR, Whisper, Gemini SDK (skips CV libraries).

## 📂 Project Structure

```
smart-notes-generator/
├── 🎯 CORE SCRIPTS
│   ├── src/process_new_lecture.py       # [START HERE] Main pipeline
│   ├── src/production_note_maker.py     # Notes generation
│   └── run_production_notes.py          # Alias for production_note_maker.py
│
├── 🔧 UTILITIES
│   ├── rerun_deduplication.py           # Instant SSIM re-optimization
│   ├── process_complete.bat/.sh         # Convenience wrappers
│   ├── generate_notes.bat/.sh           # Quick notes generation
│   ├── test_paddle.py                   # Verify OCR installation
│   ├── reprocess_all.bat/.ps1           # Batch process all videos
│   ├── install_notes_deps.ps1           # Minimal dependency install
│   └── src/generate_final_preview.py    # HTML preview generator
│
├── 🧪 ANALYSIS & DEVELOPMENT
│   ├── compare_deduplication.py         # Algorithm comparison
│   ├── extract_with_clustering_only.py  # Clustering analysis
│   ├── src/visualize_transitions.py     # HTML preview generator
│   ├── src/train_xgboost_model.py       # Model training
│   ├── src/merge_and_split_dataset.py   # Dataset preparation
│   └── src/compare_transitions.py       # Ground truth comparison
│
├── 📊 DATA STRUCTURE
│   ├── data/
│   │   ├── videos/                      # [PUT VIDEOS HERE]
│   │   ├── lectures/<video_id>/         # Output per video
│   │   │   ├── slides/*.png             # Extracted slides
│   │   │   ├── audio/*.wav              # Audio segments (optional)
│   │   │   ├── transitions.json         # Metadata + timestamps
│   │   │   ├── ocr_cache.json           # OCR cache (auto-generated)
│   │   │   ├── transcript_cache.json    # Whisper cache (auto-generated)
│   │   │   └── notes.md                 # Final notes
│   │   ├── master_dataset.csv           # Training data (all videos)
│   │   └── train_dataset.csv / test_dataset.csv
│   └── models/
│       └── xgboost_transition_classifier_*.json  # Trained model
│
├── 📖 DOCUMENTATION
│   ├── README.md                        # This file
│   ├── setup_github.md                  # Git workflow guide
│   ├── PROJECT_HISTORY_MASTER.md        # Complete development history
│   ├── COMPARISON_GUIDE.md              # Algorithm analysis
│   └── docs/README.md                   # Additional docs
│
└── ⚙️ CONFIGURATION
    ├── requirements.txt                 # Python dependencies
    ├── .gitignore                       # Git exclusions
    ├── .env.example                     # Environment template
    └── LICENSE                          # MIT License
```

## 🔧 Configuration

### SSIM Deduplication Thresholds

- **0.85**: Strict (for incremental board writing) - **Recommended for deadlock/OS videos**
- **0.92**: Moderate (balanced approach)
- **0.95**: Permissive (for discrete slide changes) - **Default**

### API Limits

- **Gemini 2.5 Flash (Free)**: 20 requests/day
- **Caching**: OCR and transcription cached for instant re-runs

## 🧪 Testing

Tested on 19+ educational videos:
- **Physics Wallah** lectures (Physics, Chemistry, Math)
- **Operating Systems** (deadlock, semaphores)
- **DSA** (Pankaj Sir)
- **Computer Networks**, **Database Management**, **TOC**

## 📊 Performance

- **Model Accuracy**: 98.68% F1-score on test set
- **Deduplication**: 30-50% slide reduction on incremental videos
- **Processing Speed**: 
  - Feature extraction: ~5 FPS
  - Slide extraction: 30 FPS for quality
  - Re-deduplication: 3 seconds (vs 40 minutes full reprocess)

## 🛠️ Hardware Requirements

### Minimum (CPU)
- 8GB RAM
- 4 cores
- 10GB disk space per lecture

### Recommended (GPU)
- 16GB RAM
- NVIDIA GPU with 4GB+ VRAM (for Whisper transcription speedup)
- 20GB disk space

## 📝 Output Format

Generated notes include:
- **Slide Screenshots**: High-quality PNG images
- **OCR Text**: Extracted text from each slide
- **Transcription**: Word-level timestamped audio
- **AI Notes**: Structured notes with:
  - Key concepts
  - Formulas (LaTeX formatted)
  - Code examples
  - Tables
  - Executive summary

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly on different video types
4. Submit pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **PaddleOCR**: OCR engine
- **OpenAI Whisper**: Audio transcription
- **Google Gemini**: Multimodal AI
- **XGBoost**: Transition detection model
- **Physics Wallah**: Test video source

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This project is optimized for educational lecture videos. Performance may vary for other video types.
