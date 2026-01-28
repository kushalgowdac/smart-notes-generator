# GitHub Setup Guide

## Step 1: Configure Git (First Time Only)

```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config user.name
git config user.email
```

## Step 2: Create Initial Commit

```bash
# Files are already staged with 'git add .'
# Now commit them
git commit -m "Initial commit: Smart Notes Generator v3.0

- Complete CV pipeline with XGBoost model (98.68% accuracy)
- Multimodal notes generation (OCR + Whisper + Gemini)
- Instant re-deduplication tool (3s vs 40min optimization)
- Smart caching system for zero-cost re-runs
- Tested on 19+ educational lecture videos
- Production ready with comprehensive documentation"
```

## Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `smart-notes-generator`
3. Description: `Automated lecture notes generator using computer vision and multimodal AI`
4. **Keep it PUBLIC** (so your friend can clone it)
5. **DO NOT** initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

## Step 4: Link Local Repository to GitHub

```bash
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/smart-notes-generator.git

# Verify remote was added
git remote -v
```

## Step 5: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 6: Share with Your Friend

Send them this link:
```
https://github.com/YOUR_USERNAME/smart-notes-generator
```

They can clone it with:
```bash
git clone https://github.com/YOUR_USERNAME/smart-notes-generator.git
cd smart-notes-generator
```

---

## Quick Commands Reference

### After making changes:
```bash
git status                    # See what changed
git add .                     # Stage all changes
git commit -m "Description"   # Commit changes
git push                      # Push to GitHub
```

### Pull latest changes (for your friend):
```bash
git pull                      # Get latest updates
```

### Check current status:
```bash
git status                    # Working directory status
git log --oneline -5          # Last 5 commits
git remote -v                 # View remote repositories
```

---

## Troubleshooting

### If Git asks for credentials:
1. **Option 1 - Personal Access Token (Recommended)**
   - Go to GitHub Settings → Developer Settings → Personal Access Tokens
   - Generate new token (classic) with `repo` scope
   - Use token as password when Git prompts

2. **Option 2 - SSH Keys**
   ```bash
   # Generate SSH key
   ssh-keygen -t ed25519 -C "your.email@example.com"
   
   # Add to GitHub: Settings → SSH and GPG keys
   # Change remote to SSH
   git remote set-url origin git@github.com:YOUR_USERNAME/smart-notes-generator.git
   ```

### If commit fails with "Please tell me who you are":
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### If you need to undo last commit (before push):
```bash
git reset --soft HEAD~1       # Keep changes staged
git reset --hard HEAD~1       # Discard changes (careful!)
```

---

## Your Friend's Setup (After Cloning)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/smart-notes-generator.git
cd smart-notes-generator

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key (if using notes generation)
copy .env.example .env
# Edit .env and add GEMINI_API_KEY

# Test on new video
python src/process_new_lecture.py data/videos/test_video.mp4
```

---

**Note**: The `.gitignore` file is already configured to exclude:
- Large video files
- Generated outputs (slides, audio, notes)
- Virtual environment files
- Cache files
- Backup files

Your friend will need to add their own lecture videos to `data/videos/` after cloning.
