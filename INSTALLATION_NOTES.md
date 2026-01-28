# Installation Notes - Notes Generation System

## ✅ Successfully Installed (January 28, 2026)

Core dependencies for lecture notes generation:
- `paddleocr>=2.7.0` - OCR text extraction
- `google-generativeai>=0.8.0` - Gemini LLM API
- `faster-whisper>=1.0.0` - Audio transcription
- `markdown>=3.5.0` - Markdown processing

## Dependency Conflict Resolution

**Issue:** grpcio version conflicts with google-api-core

**Solution:** Updated requirements.txt with:
- Explicit grpcio version pinning: `grpcio>=1.60.0,<2.0.0`
- Compatible google-generativeai version: `>=0.8.0`

## Optional: PDF Generation

PDF generation requires `weasyprint` which has additional system dependencies.

**To install (optional):**
```cmd
pip install weasyprint
```

**Note:** If weasyprint fails to install, notes will still generate in Markdown and JSON formats. PDF is optional.

## Next Steps

1. **Set Google API Key:**
   ```cmd
   set GOOGLE_API_KEY=your-key-here
   ```
   Get free key at: https://aistudio.google.com/app/apikey

2. **Test the system:**
   ```cmd
   generate_notes.bat data/lectures/dsa_pankaj_sir
   ```

3. **Check output:**
   - `data/lectures/dsa_pankaj_sir/lecture_notes.md`
   - `data/lectures/dsa_pankaj_sir/notes_data.json`
   - `data/lectures/dsa_pankaj_sir/lecture_notes.pdf` (if weasyprint installed)

## Troubleshooting

If you encounter import errors:
```cmd
pip install --upgrade paddleocr google-generativeai faster-whisper markdown
```

If grpcio errors persist:
```cmd
pip uninstall grpcio grpcio-status
pip install grpcio==1.60.0 grpcio-status==1.60.0
pip install google-generativeai
```

## System Ready ✨

All core dependencies are installed. You can now generate lecture notes!
