# Lecture: algo_1

## Folder Structure

```
algo_1/
├── transitions.json          - All deduplicated transition timestamps (including blanks)
├── transition_previews/      - Preview images of all transitions
│   ├── transition_001.jpg    - First transition frame
│   ├── transition_002.jpg    - Second transition frame
│   └── ...
├── slides/                   - (Future) Best quality slide images
│   ├── slide_001.png         - High-res slide 1
│   ├── slide_002.png         - High-res slide 2
│   └── ...
├── audio/                    - (Future) Extracted audio segments
│   ├── slide_001.mp3         - Audio for slide 1
│   ├── slide_002.mp3         - Audio for slide 2
│   └── ...
├── metadata.json            - (Future) Best slide timestamps and quality metrics
└── README.md                - This file
```

## Current Status

- **Total Transitions:** 10
- **Blank Slides:** 0
- **Content Slides:** 10

## Files Created

✅ `transitions.json` - Complete transition data with timestamps
✅ `transition_previews/` - All transition frame images (640px width)
⏳ `slides/` - Awaiting high-resolution extraction
⏳ `audio/` - Awaiting FFmpeg audio extraction
⏳ `metadata.json` - Awaiting quality analysis

## Next Steps

1. **Extract Best Slides:**
   - Use `transitions.json` to identify content slides (is_blank=false)
   - Extract high-resolution frames at exact timestamps
   - Apply quality enhancement if needed
   - Save to `slides/` folder

2. **Extract Audio:**
   - Use audio_window start/end from `transitions.json`
   - Run FFmpeg to extract audio segments
   - Save to `audio/` folder

3. **Create Metadata:**
   - Analyze slide quality (sharpness, contrast, text clarity)
   - Add OCR text extraction results
   - Add multimodal AI analysis results
   - Save to `metadata.json`

4. **Multimodal AI Integration:**
   - Feed slide image + audio to GPT-4 Vision / Gemini
   - Generate structured lecture notes
   - Save as `lecture_notes.md` or `notes.json`

## Transition Data Format

```json
{
  "index": 1,
  "timestamp": 123.45,
  "timestamp_formatted": "02:03",
  "is_blank": false,
  "ssim_score": 0.85,
  "edge_count": 0.75,
  "has_audio": true,
  "audio_window": {
    "start": 120.0,
    "end": 145.5,
    "duration": 25.5
  },
  "preview_image": "transition_previews/transition_001.jpg",
  "notes": "Optional manual annotations"
}
```

## Usage

### Load Transitions
```python
import json

with open('transitions.json', 'r') as f:
    data = json.load(f)

# Get all content transitions (exclude blanks)
content_transitions = [
    t for t in data['transitions'] 
    if not t['is_blank']
]

# Get timestamps for extraction
timestamps = [t['timestamp'] for t in content_transitions]
```

### Extract High-Res Slides
```python
import cv2

cap = cv2.VideoCapture('../videos/algo_1.mp4')

for trans in content_transitions:
    timestamp = trans['timestamp']
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    
    if ret:
        output_path = f"slides/slide_{trans['index']:03d}.png"
        cv2.imwrite(output_path, frame)

cap.release()
```

### Extract Audio with FFmpeg
```bash
# Read audio_window from transitions.json
# For each transition:
ffmpeg -i ../videos/algo_1.mp4 \
    -ss <audio_window.start> \
    -to <audio_window.end> \
    -vn -acodec libmp3lame -q:a 2 \
    audio/slide_001.mp3
```

---
Generated: 2026-01-27 10:47:53
