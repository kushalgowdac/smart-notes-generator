# dsa_pankaj_sir

## Video Information
- **Duration**: 15:50 (950.25s)
- **Total Transitions Detected**: 20
- **Content Slides**: 10
- **Blank Slides**: 10
- **Processed**: 2026-01-28 07:40:53

## Pipeline Details
- **Model**: xgboost_model_20260126_160645.pkl
- **Prediction Threshold**: 0.01
- **SSIM Dedup Threshold**: 0.95
- **F1-Score**: 98.68% (on test set)

## Output Structure
```
dsa_pankaj_sir/
├── audio/                    # Extracted audio WAV
├── slides/                   # Best quality slide PNGs
├── transition_previews/      # Preview JPEGs (640px)
├── transitions.json          # Deduplicated timestamps + metadata
├── metadata.json             # Slide extraction details
└── README.md                 # This file
```

## Usage

### Extract Audio Segment
```python
import json
with open('transitions.json') as f:
    data = json.load(f)
    
# Get audio for slide 1
slide_1 = data['transitions'][0]
start = slide_1['audio_window']['start']
end = slide_1['audio_window']['end']

# Use ffmpeg
ffmpeg -i audio/dsa_pankaj_sir.wav -ss {start} -to {end} slide_1_audio.wav
```

### View Slide with Timestamp
```python
# Load metadata
with open('metadata.json') as f:
    meta = json.load(f)
    
for slide in meta['slides']:
    print(f"Slide {slide['slide_number']}: {slide['filename']}")
    print(f"  Captured at: {slide['capture_timestamp']}s")
    print(f"  Quality Score: {slide['scoring']['quality_score']:.3f}")
```

## Slide Quality Metrics
Each slide is scored on:
- **Board Visibility** (blackboard/whiteboard/greenboard detection)
- **Edge Density** (text/chalk clarity)
- **Distribution** (text spread across frame)
- **Sharpness** (motion blur detection)
- **Brightness** (exposure quality)
- **Proximity** (position in adaptive window)

## Next Steps
1. Review slides in `slides/` folder
2. Extract audio segments for each slide
3. Use multimodal AI (GPT-4 Vision/Gemini) for note generation
4. Combine slide images + audio + AI notes into final document
