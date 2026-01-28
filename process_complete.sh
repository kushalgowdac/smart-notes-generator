#!/bin/bash
# Complete lecture processing pipeline - slides + notes generation
# Usage: ./process_complete.sh path/to/video.mp4 [--skip-notes]

echo "======================================================================"
echo "SMART NOTES GENERATOR - COMPLETE PIPELINE"
echo "======================================================================"
echo

if [ -z "$1" ]; then
    echo "Error: No video file specified"
    echo "Usage: ./process_complete.sh path/to/video.mp4 [--skip-notes]"
    exit 1
fi

VIDEO_PATH="$1"
SKIP_NOTES="$2"
VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')

echo "Video: $VIDEO_PATH"
echo "Output: data/lectures/$VIDEO_NAME"
echo

# Step 1: Extract slides and transitions
echo "[1/2] Extracting slides and transitions..."
echo "----------------------------------------------------------------------"
.venv/Scripts/python.exe src/process_new_lecture.py "$VIDEO_PATH"

if [ $? -ne 0 ]; then
    echo
    echo "Error: Slide extraction failed!"
    exit 1
fi

echo
echo "✓ Slide extraction complete!"
echo

# Step 2: Generate lecture notes (unless --skip-notes)
if [ "$SKIP_NOTES" = "--skip-notes" ]; then
    echo "Skipping notes generation (--skip-notes flag)"
    echo
    echo "======================================================================"
    echo "PIPELINE COMPLETE (SLIDES ONLY)"
    echo "======================================================================"
    echo "Output: data/lectures/$VIDEO_NAME"
    echo
    echo "Next steps:"
    echo "  - Review slides in data/lectures/$VIDEO_NAME/slides/"
    echo "  - Generate notes: python src/generate_lecture_notes.py data/lectures/$VIDEO_NAME"
    exit 0
fi

echo "[2/2] Generating lecture notes with AI..."
echo "----------------------------------------------------------------------"
.venv/Scripts/python.exe src/generate_lecture_notes.py "data/lectures/$VIDEO_NAME"

if [ $? -ne 0 ]; then
    echo
    echo "Warning: Notes generation failed (slides are still available)"
    exit 1
fi

echo
echo "======================================================================"
echo "✓ COMPLETE PIPELINE FINISHED!"
echo "======================================================================"
echo "Output directory: data/lectures/$VIDEO_NAME"
echo
echo "Generated files:"
echo "  - slides/*.png            (extracted slide images)"
echo "  - audio/audio.wav         (extracted audio)"
echo "  - transitions.json        (transition timestamps)"
echo "  - lecture_notes.md        (formatted study notes)"
echo "  - notes_data.json         (structured notes data)"
echo "  - metadata.json           (slide metadata)"
echo
