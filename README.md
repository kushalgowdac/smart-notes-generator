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

#### Process New Lecture Video

```bash
# Full pipeline (slides + notes)
python src/process_new_lecture.py data/videos/my_lecture.mp4

# Slides only (skip notes generation)
python src/process_new_lecture.py data/videos/my_lecture.mp4 --skip-notes
```

#### Generate Notes from Existing Slides

```bash
python src/production_note_maker.py data/lectures/my_lecture
```

#### Re-optimize Deduplication

```bash
# Adjust SSIM threshold without reprocessing video
python rerun_deduplication.py data/lectures/my_lecture --ssim 0.85
```

## 📂 Project Structure

```
smart-notes-generator/
├── src/
│   ├── process_new_lecture.py       # Main CV pipeline
│   ├── production_note_maker.py     # Notes generation
│   ├── video_feature_extractor.py   # Feature extraction (5 FPS)
│   ├── train_xgboost_model.py       # Model training
│   └── ...
├── data/
│   ├── videos/                      # Input videos
│   ├── lectures/                    # Per-video outputs
│   │   └── <video_id>/
│   │       ├── slides/              # Extracted slides
│   │       ├── audio/               # Audio segments
│   │       ├── transitions.json     # Metadata
│   │       └── notes.md             # Generated notes
│   └── ...
├── models/                          # Trained XGBoost model
├── rerun_deduplication.py           # Instant re-optimization tool
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
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
