@echo off
REM Smart Notes Generator - Lecture Notes Generation Helper
REM Usage: generate_notes.bat <lecture_folder>
REM Example: generate_notes.bat data/lectures/dsa_pankaj_sir

echo ================================================
echo Smart Notes Generator - Notes Generation
echo ================================================
echo.

if "%1"=="" (
    echo ERROR: No lecture folder specified
    echo.
    echo Usage: generate_notes.bat ^<lecture_folder^>
    echo Example: generate_notes.bat data/lectures/dsa_pankaj_sir
    echo.
    pause
    exit /b 1
)

REM Check if folder exists
if not exist "%1" (
    echo ERROR: Folder not found: %1
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please create virtual environment first: python -m venv .venv
    echo.
    pause
    exit /b 1
)

REM Check for API key
if "%GOOGLE_API_KEY%"=="" (
    echo WARNING: GOOGLE_API_KEY environment variable not set
    echo.
    echo Get your free API key at: https://aistudio.google.com/app/apikey
    echo.
    echo Set it using:
    echo   set GOOGLE_API_KEY=your-key-here
    echo.
    echo Or pass as argument:
    echo   generate_notes.bat %1 YOUR_API_KEY
    echo.
    
    if not "%2"=="" (
        echo Using API key from command line argument...
        set GOOGLE_API_KEY=%2
    ) else (
        set /p GOOGLE_API_KEY="Enter your Google API key (or press Enter to exit): "
        if "!GOOGLE_API_KEY!"=="" (
            echo No API key provided. Exiting.
            pause
            exit /b 1
        )
    )
)

echo.
echo Starting notes generation...
echo Lecture: %1
echo.

REM Run the script
".venv\Scripts\python.exe" src/generate_lecture_notes.py "%1"

echo.
echo ================================================
echo Notes generation complete!
echo ================================================
echo.
pause
