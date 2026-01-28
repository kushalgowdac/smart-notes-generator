# Lecture Notes Generation Guide

## Overview

The Smart Notes Generator now includes an intelligent notes generation system that creates comprehensive lecture notes from your processed slides and audio. The system supports **Hindi/English audio** and generates notes in **English**.

## Features

- **Multi-language Support**: Automatically detects Hindi/English audio and translates to English notes
- **Advanced OCR**: PaddleOCR with preprocessing for better text extraction
- **Audio Transcription**: Faster-Whisper (Large-v3) for accurate transcription
- **LLM-based Notes**: Google Gemini 1.5 Flash (Free) for intelligent note generation
- **Multiple Formats**: Generates Markdown, JSON, and PDF outputs

## Prerequisites

### 1. Install Dependencies

```bash
# Activate your virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install new dependencies
pip install paddleocr google-generativeai faster-whisper markdown weasyprint
```

Or update from requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Get Google API Key (Free)

1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key

**Set as environment variable:**

Windows:
```cmd
set GOOGLE_API_KEY=your-api-key-here
```

Linux/Mac:
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

Or add to your `.env` file or system environment variables permanently.

## Usage

### Method 1: Using Helper Script (Recommended)

**Windows:**
```cmd
generate_notes.bat data/lectures/dsa_pankaj_sir
```

**Linux/Mac:**
```bash
chmod +x generate_notes.sh
./generate_notes.sh data/lectures/dsa_pankaj_sir
```

**With API key as argument:**
```cmd
generate_notes.bat data/lectures/dsa_pankaj_sir YOUR_API_KEY
```

### Method 2: Direct Python Execution

```bash
python src/generate_lecture_notes.py data/lectures/dsa_pankaj_sir
```

With API key:
```bash
python src/generate_lecture_notes.py data/lectures/dsa_pankaj_sir --api-key YOUR_API_KEY
```

## Input Requirements

The lecture folder must have been processed by `process_new_lecture.py` first. Required structure:

```
data/lectures/dsa_pankaj_sir/
├── slides/              # Required: Extracted slide images
│   ├── slide_001.jpg
│   ├── slide_002.jpg
│   └── ...
├── audio/               # Required: Audio file
│   └── dsa_pankaj_sir.wav
├── transitions.json     # Recommended: For accurate timing
└── metadata.json        # Optional: Lecture metadata
```

## Output Files

After running notes generation, you'll get:

```
data/lectures/dsa_pankaj_sir/
├── lecture_notes.md     # ✨ Comprehensive Markdown notes
├── lecture_notes.pdf    # ✨ PDF version (if weasyprint installed)
└── notes_data.json      # ✨ Structured JSON data
```

### Sample Output Structure

**lecture_notes.md:**
- Executive Summary
- Table of Contents (with timestamps)
- Detailed notes for each slide:
  - Slide image
  - OCR-extracted text (cleaned)
  - Teacher's explanation (transcribed)
  - LLM-generated structured notes
- Generation details

**notes_data.json:**
- Raw OCR text
- Cleaned OCR text
- Transcript segments
- Generated notes
- Timestamps and metadata

## Configuration

Edit `Config` class in [src/generate_lecture_notes.py](src/generate_lecture_notes.py):

```python
class Config:
    # Model settings
    GEMINI_MODEL = 'gemini-1.5-flash'  # Free tier
    WHISPER_MODEL = 'large-v3'         # Best for Hindi/English
    
    # Language settings
    AUDIO_LANGUAGE = 'auto'  # 'auto', 'hi', 'en'
    OUTPUT_LANGUAGE = 'en'   # Always English notes
    
    # Processing settings
    RATE_LIMIT_DELAY = 2  # seconds between API calls
```

## Processing Pipeline

1. **OCR Extraction**: Extract text from slide images with preprocessing
2. **Audio Transcription**: Transcribe audio with timestamps using Whisper
3. **Alignment**: Match transcript segments to slide timestamps
4. **OCR Cleaning**: Use LLM to fix OCR errors using audio context
5. **Notes Generation**: Create structured notes combining slide + audio

## Tips for Best Results

1. **Audio Quality**: Clear audio improves transcription accuracy
2. **Slide Quality**: High-resolution slides improve OCR accuracy
3. **API Key**: Use environment variable for security
4. **Rate Limiting**: Free tier has limits (2 seconds between calls)
5. **PDF Generation**: Install weasyprint for PDF output (optional)

## Troubleshooting

### "PaddleOCR not installed"
```bash
pip install paddleocr
```

### "Google API key required"
Set environment variable:
```cmd
set GOOGLE_API_KEY=your-key-here
```

### "faster-whisper not installed"
```bash
pip install faster-whisper
```

### PDF generation fails
PDF is optional. Install weasyprint:
```bash
pip install weasyprint
```

### Out of memory during Whisper
Use smaller model in Config:
```python
WHISPER_MODEL = 'base'  # or 'small', 'medium'
```

## Example Workflow

Complete workflow from video to notes:

```bash
# Step 1: Process video (extract slides and audio)
python src/process_new_lecture.py data/videos/dsa_pankaj_sir.mp4

# Step 2: Generate notes
generate_notes.bat data/lectures/dsa_pankaj_sir

# Output: lecture_notes.md, lecture_notes.pdf, notes_data.json
```

## Batch Processing

Process multiple lectures:

**Windows (create `generate_all_notes.bat`):**
```cmd
@echo off
for /d %%D in (data\lectures\*) do (
    echo Processing %%D...
    generate_notes.bat %%D
)
```

**Linux/Mac (create `generate_all_notes.sh`):**
```bash
#!/bin/bash
for dir in data/lectures/*/; do
    echo "Processing $dir..."
    ./generate_notes.sh "$dir"
done
```

## API Cost (Free Tier)

Google Gemini 1.5 Flash Free Tier:
- **15 requests per minute**
- **1,500 requests per day**
- **1 million tokens per day**

With `RATE_LIMIT_DELAY=2` seconds:
- Can process ~30 slides per minute
- ~450 slides per day (well within limits)

## Language Support

| Audio Language | Detection | Output Language |
|----------------|-----------|-----------------|
| English        | Auto      | English         |
| Hindi          | Auto      | English         |
| Mixed          | Auto      | English         |
| Other          | Manual    | English         |

To manually set language:
```python
Config.AUDIO_LANGUAGE = 'hi'  # Force Hindi
```

## Advanced Usage

### Custom Prompts

Edit prompts in `FreeGeminiLLM` class methods:
- `clean_ocr()`: OCR cleaning prompt
- `generate_notes()`: Notes generation prompt
- `generate_summary()`: Summary generation prompt

### Custom Styling

Edit PDF CSS in `create_pdf()` method for custom styling.

## Support

For issues or questions:
1. Check [README.md](README.md) for general setup
2. Review [UPDATE_GUIDE_v1.1.0.md](UPDATE_GUIDE_v1.1.0.md) for version updates
3. Ensure all dependencies are installed
4. Verify API key is set correctly

---

**Happy Note-Taking! 📚✨**
