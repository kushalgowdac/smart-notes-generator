@echo off
REM Quick script to re-process all videos with new features
REM Version 1.1.0 - Skin Detection & Teacher Presence

echo ========================================
echo Smart Notes Generator - Re-processing
echo Version 1.1.0 Update
echo ========================================
echo.
echo This will:
echo  - Backup existing CSV files
echo  - Re-process all 19 videos
echo  - Add skin_pixel_ratio feature
echo  - Add teacher_presence feature
echo.
echo Estimated time: 15-17 hours
echo.
pause

echo.
echo Starting re-processing...
python src\reprocess_videos.py

echo.
echo ========================================
echo Re-processing Complete!
echo ========================================
echo.
echo New features added:
echo  - skin_pixel_ratio
echo  - teacher_presence
echo.
echo Total features: 25 (was 23)
echo.
pause
