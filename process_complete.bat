@echo off
REM Complete lecture processing pipeline - slides + notes generation
REM Usage: process_complete.bat path/to/video.mp4 [--skip-notes]

echo ======================================================================
echo SMART NOTES GENERATOR - COMPLETE PIPELINE
echo ======================================================================
echo.

if "%~1"=="" (
    echo Error: No video file specified
    echo Usage: process_complete.bat path/to/video.mp4 [--skip-notes]
    exit /b 1
)

set VIDEO_PATH=%~1
set SKIP_NOTES=%~2

REM Get video name without extension
for %%F in ("%VIDEO_PATH%") do set VIDEO_NAME=%%~nF

echo Video: %VIDEO_PATH%
echo Output: data/lectures/%VIDEO_NAME%
echo.

REM Step 1: Extract slides and transitions
echo [1/2] Extracting slides and transitions...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe src\process_new_lecture.py "%VIDEO_PATH%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Slide extraction failed!
    exit /b 1
)

echo.
echo ✓ Slide extraction complete!
echo.

REM Step 2: Generate lecture notes (unless --skip-notes)
if "%SKIP_NOTES%"=="--skip-notes" (
    echo Skipping notes generation (--skip-notes flag)
    echo.
    echo ======================================================================
    echo PIPELINE COMPLETE (SLIDES ONLY)
    echo ======================================================================
    echo Output: data/lectures/%VIDEO_NAME%
    echo.
    echo Next steps:
    echo   - Review slides in data/lectures/%VIDEO_NAME%/slides/
    echo   - Generate notes: python src\generate_lecture_notes.py data/lectures/%VIDEO_NAME%
    exit /b 0
)

echo [2/2] Generating lecture notes with AI...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe src\generate_lecture_notes.py "data\lectures\%VIDEO_NAME%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Warning: Notes generation failed (slides are still available)
    exit /b 1
)

echo.
echo ======================================================================
echo ✓ COMPLETE PIPELINE FINISHED!
echo ======================================================================
echo Output directory: data/lectures/%VIDEO_NAME%
echo.
echo Generated files:
echo   - slides/*.png            (extracted slide images)
echo   - audio/audio.wav         (extracted audio)
echo   - transitions.json        (transition timestamps)
echo   - lecture_notes.md        (formatted study notes)
echo   - notes_data.json         (structured notes data)
echo   - metadata.json           (slide metadata)
echo.
