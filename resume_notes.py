"""
Resume Notes Generation - Smart checkpoint recovery
Resumes notes generation from where it left off when API quota runs out

Usage:
    python resume_notes.py data/lectures/my_lecture
    python resume_notes.py data/lectures/my_lecture --start-from 18
    python resume_notes.py data/lectures/my_lecture --force-regenerate
"""

import sys
import json
import re
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_last_completed_slide(notes_file):
    """Parse notes.md and find the last fully completed slide"""
    if not notes_file.exists():
        logger.info("No existing notes.md found - starting from beginning")
        return 0
    
    with open(notes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all slide headers (## Slide X: ...)
    slide_pattern = r'^## Slide (\d+):'
    matches = list(re.finditer(slide_pattern, content, re.MULTILINE))
    
    if not matches:
        logger.info("No slides found in notes.md - starting from beginning")
        return 0
    
    # Get last slide number
    last_slide = int(matches[-1].group(1))
    
    # Check if last slide has content after it (not just header)
    last_match_pos = matches[-1].end()
    remaining_content = content[last_match_pos:].strip()
    
    # If next section starts immediately or no content, last slide is incomplete
    if not remaining_content or remaining_content.startswith('##') or len(remaining_content) < 50:
        logger.warning(f"Slide {last_slide} appears incomplete (no content or too short)")
        return last_slide - 1 if last_slide > 1 else 0
    
    logger.info(f"Last completed slide: {last_slide}")
    return last_slide


def get_total_slides(lecture_dir):
    """Get total number of slides from transitions.json"""
    transitions_file = lecture_dir / 'transitions.json'
    
    if not transitions_file.exists():
        logger.error(f"transitions.json not found in {lecture_dir}")
        return None
    
    with open(transitions_file, 'r') as f:
        data = json.load(f)
    
    return data['total_transitions']


def check_caches_exist(lecture_dir):
    """Verify OCR and transcription caches exist"""
    ocr_cache = lecture_dir / 'ocr_cache.json'
    transcript_cache = lecture_dir / 'transcript_cache.json'
    
    status = {
        'ocr': ocr_cache.exists(),
        'transcript': transcript_cache.exists()
    }
    
    return status


def main():
    parser = argparse.ArgumentParser(
        description='Resume notes generation from checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect last completed slide and resume
  python resume_notes.py data/lectures/deadlock_os
  
  # Start from specific slide number
  python resume_notes.py data/lectures/deadlock_os --start-from 18
  
  # Force regenerate all (ignore checkpoint)
  python resume_notes.py data/lectures/deadlock_os --force-regenerate
  
  # Dry run (show what would be done)
  python resume_notes.py data/lectures/deadlock_os --dry-run
        """
    )
    
    parser.add_argument('lecture_dir', help='Path to lecture directory')
    parser.add_argument('--start-from', type=int, help='Start from specific slide number (overrides auto-detection)')
    parser.add_argument('--force-regenerate', action='store_true', help='Ignore checkpoint and regenerate all notes')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    
    args = parser.parse_args()
    
    lecture_path = Path(args.lecture_dir)
    
    if not lecture_path.exists():
        logger.error(f"Lecture directory not found: {lecture_path}")
        return 1
    
    logger.info("="*70)
    logger.info(f"RESUME NOTES GENERATION: {lecture_path.name}")
    logger.info("="*70)
    
    # Get total slides
    total_slides = get_total_slides(lecture_path)
    if total_slides is None:
        return 1
    
    logger.info(f"Total slides in video: {total_slides}")
    
    # Determine starting point
    if args.force_regenerate:
        start_slide = 1
        logger.info("FORCE REGENERATE: Starting from slide 1")
    elif args.start_from:
        start_slide = args.start_from
        logger.info(f"MANUAL START: Starting from slide {start_slide}")
    else:
        notes_file = lecture_path / 'notes.md'
        last_completed = detect_last_completed_slide(notes_file)
        start_slide = last_completed + 1
        logger.info(f"AUTO-DETECTED: Resuming from slide {start_slide}")
    
    # Validate range
    if start_slide < 1:
        start_slide = 1
    if start_slide > total_slides:
        logger.info(f"✓ All slides already completed ({total_slides}/{total_slides})")
        return 0
    
    slides_to_process = total_slides - start_slide + 1
    logger.info(f"Slides to process: {slides_to_process} (slides {start_slide}-{total_slides})")
    
    # Check cache status
    logger.info("")
    logger.info("Cache status:")
    cache_status = check_caches_exist(lecture_path)
    logger.info(f"  OCR cache: {'✓ EXISTS' if cache_status['ocr'] else '✗ MISSING (will regenerate)'}")
    logger.info(f"  Transcript cache: {'✓ EXISTS' if cache_status['transcript'] else '✗ MISSING (will regenerate)'}")
    
    # API quota warning
    api_calls_needed = slides_to_process
    if not cache_status['ocr']:
        api_calls_needed += slides_to_process * 0  # OCR is free
    
    logger.info("")
    logger.info(f"⚠️  Estimated API calls needed: {api_calls_needed} (Gemini)")
    logger.info(f"   Free tier quota: 20 requests/day")
    
    if api_calls_needed > 20:
        logger.warning(f"   This will require ~{(api_calls_needed + 19) // 20} days to complete")
    
    # Dry run exit
    if args.dry_run:
        logger.info("")
        logger.info("="*70)
        logger.info("DRY RUN COMPLETE - No changes made")
        logger.info("="*70)
        return 0
    
    # Build command
    logger.info("")
    logger.info("="*70)
    logger.info("EXECUTING NOTES GENERATION")
    logger.info("="*70)
    
    # Import and run production_note_maker with range
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    try:
        from production_note_maker import NoteMaker
        
        note_maker = NoteMaker(str(lecture_path))
        
        # Modify to support slide range
        logger.info(f"Processing slides {start_slide} to {total_slides}...")
        
        # Create custom range processing
        success = note_maker.generate_notes(
            start_slide=start_slide,
            end_slide=total_slides,
            append_mode=(start_slide > 1)  # Append if resuming
        )
        
        if success:
            logger.info("")
            logger.info("="*70)
            logger.info("✓ NOTES GENERATION COMPLETED")
            logger.info("="*70)
            logger.info(f"Output: {lecture_path / 'notes.md'}")
            return 0
        else:
            logger.error("Notes generation failed")
            return 1
            
    except ImportError as e:
        logger.error(f"Could not import production_note_maker: {e}")
        logger.info("")
        logger.info("FALLBACK: Run this command manually:")
        logger.info(f"  python src/production_note_maker.py {lecture_path}")
        logger.info(f"  Then manually edit to skip slides 1-{start_slide-1}")
        return 1
    except AttributeError:
        # production_note_maker doesn't support range yet
        logger.warning("production_note_maker.py doesn't support slide range yet")
        logger.info("")
        logger.info("WORKAROUND:")
        logger.info(f"1. Run: python src/production_note_maker.py {lecture_path}")
        logger.info(f"2. It will use cached OCR/transcript (zero cost)")
        logger.info(f"3. Manually skip slides 1-{start_slide-1} in the output")
        logger.info(f"4. Or wait for API quota reset and process slides {start_slide}-{total_slides}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
