# Install dependencies for notes generation
Write-Host "Installing dependencies for lecture notes generation..." -ForegroundColor Cyan

$pythonExe = "D:\College_Life\projects\smart notes generator - trail 3\.venv\Scripts\python.exe"

Write-Host "`nUsing Python: $pythonExe" -ForegroundColor Yellow

# Install using uv
Write-Host "`nInstalling packages..." -ForegroundColor Green
uv pip install paddleocr google-generativeai faster-whisper markdown --python $pythonExe

Write-Host "`nVerifying installation..." -ForegroundColor Green
& $pythonExe -c "import paddleocr; import google.generativeai; import faster_whisper; import markdown; print('All packages installed successfully!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Installation complete!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Installation failed. Please check errors above." -ForegroundColor Red
}
