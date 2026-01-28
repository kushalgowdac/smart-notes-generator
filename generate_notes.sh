#!/bin/bash
# Smart Notes Generator - Lecture Notes Generation Helper
# Usage: ./generate_notes.sh <lecture_folder>
# Example: ./generate_notes.sh data/lectures/dsa_pankaj_sir

echo "================================================"
echo "Smart Notes Generator - Notes Generation"
echo "================================================"
echo ""

if [ -z "$1" ]; then
    echo "ERROR: No lecture folder specified"
    echo ""
    echo "Usage: ./generate_notes.sh <lecture_folder>"
    echo "Example: ./generate_notes.sh data/lectures/dsa_pankaj_sir"
    echo ""
    exit 1
fi

# Check if folder exists
if [ ! -d "$1" ]; then
    echo "ERROR: Folder not found: $1"
    echo ""
    exit 1
fi

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ] && [ ! -f ".venv/Scripts/python.exe" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please create virtual environment first: python -m venv .venv"
    echo ""
    exit 1
fi

# Determine Python path
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON=".venv/Scripts/python.exe"
fi

# Check for API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "WARNING: GOOGLE_API_KEY environment variable not set"
    echo ""
    echo "Get your free API key at: https://aistudio.google.com/app/apikey"
    echo ""
    echo "Set it using:"
    echo "  export GOOGLE_API_KEY='your-key-here'"
    echo ""
    echo "Or pass as argument:"
    echo "  ./generate_notes.sh $1 YOUR_API_KEY"
    echo ""
    
    if [ -n "$2" ]; then
        echo "Using API key from command line argument..."
        export GOOGLE_API_KEY="$2"
    else
        read -p "Enter your Google API key (or press Enter to exit): " api_key
        if [ -z "$api_key" ]; then
            echo "No API key provided. Exiting."
            exit 1
        fi
        export GOOGLE_API_KEY="$api_key"
    fi
fi

echo ""
echo "Starting notes generation..."
echo "Lecture: $1"
echo ""

# Run the script
$PYTHON src/generate_lecture_notes.py "$1"

echo ""
echo "================================================"
echo "Notes generation complete!"
echo "================================================"
echo ""
