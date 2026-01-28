# PowerShell script to label all CSVs with ground truth transitions
# Version 1.1.0

Write-Host "========================================"
Write-Host "Smart Notes Generator - Transition Labeling"
Write-Host "========================================"
Write-Host ""
Write-Host "This will:"
Write-Host "  - Load manual timestamps from ground_truth/"
Write-Host "  - Find lowest SSIM frame in +/- 1.5s window"
Write-Host "  - Label frame +/- 5 frames as transition (1)"
Write-Host "  - Update all 19 CSV files"
Write-Host ""

$confirmation = Read-Host "Continue? (Y/N)"
if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Starting labeling process..." -ForegroundColor Green
python src\label_transitions.py

Write-Host ""
Write-Host "========================================"
Write-Host "Labeling Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "Your CSV files now have:" -ForegroundColor Green
Write-Host "  - label=1 for transition frames"
Write-Host "  - label=0 for non-transition frames"
Write-Host ""
Write-Host "Ready for ML model training!" -ForegroundColor Cyan
Write-Host ""
