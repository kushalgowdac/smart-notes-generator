# PowerShell script to re-process all videos with new features
# Version 1.1.0 - Skin Detection & Teacher Presence

Write-Host "========================================"
Write-Host "Smart Notes Generator - Re-processing"
Write-Host "Version 1.1.0 Update"
Write-Host "========================================"
Write-Host ""
Write-Host "This will:"
Write-Host "  - Backup existing CSV files"
Write-Host "  - Re-process all 19 videos"
Write-Host "  - Add skin_pixel_ratio feature"
Write-Host "  - Add teacher_presence feature"
Write-Host ""
Write-Host "Estimated time: 15-17 hours" -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "Continue? (Y/N)"
if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Starting re-processing..." -ForegroundColor Green
python src\reprocess_videos.py

Write-Host ""
Write-Host "========================================"
Write-Host "Re-processing Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "New features added:" -ForegroundColor Green
Write-Host "  - skin_pixel_ratio"
Write-Host "  - teacher_presence"
Write-Host ""
Write-Host "Total features: 25 (was 23)" -ForegroundColor Cyan
Write-Host ""
