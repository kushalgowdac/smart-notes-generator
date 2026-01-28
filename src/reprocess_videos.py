"""
File: reprocess_videos.py
Purpose: Re-process all videos with updated feature set (includes skin detection & teacher presence)
Author: Smart Notes Generator Team
Created: 2026-01-26
Last Modified: 2026-01-26

This script will:
1. Backup existing CSV files
2. Re-process all videos with new features
3. Generate updated CSV files with 25 features (added skin_pixel_ratio and teacher_presence)
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video_feature_extractor import process_video_folder, logger


def backup_existing_csvs(output_dir: str = "data/output", backup_dir: str = "data/output_backup"):
    """
    Backup existing CSV files before re-processing.
    
    Args:
        output_dir: Directory containing existing CSV files
        backup_dir: Directory to store backup files
    """
    output_path = Path(output_dir)
    backup_path = Path(backup_dir)
    
    if not output_path.exists():
        logger.info(f"No existing output directory found at {output_dir}")
        return
    
    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_path / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(output_path.glob("*.csv"))
    
    if not csv_files:
        logger.info("No CSV files found to backup")
        return
    
    logger.info(f"Backing up {len(csv_files)} CSV files to {backup_path}")
    
    # Copy each CSV file to backup
    for csv_file in csv_files:
        destination = backup_path / csv_file.name
        shutil.copy2(csv_file, destination)
        logger.info(f"Backed up: {csv_file.name}")
    
    logger.info(f"Backup complete! Files saved to: {backup_path}")
    return backup_path


def main():
    """Main entry point for re-processing videos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Re-process videos with updated feature set (skin detection & teacher presence)'
    )
    parser.add_argument('--input', '-i', type=str, default='data/videos',
                       help='Input folder containing videos (default: data/videos)')
    parser.add_argument('--output', '-o', type=str, default='data/output',
                       help='Output folder for CSV files (default: data/output)')
    parser.add_argument('--fps', '-f', type=int, default=5,
                       help='Target FPS for processing (default: 5)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backing up existing CSV files')
    parser.add_argument('--backup-dir', type=str, default='data/output_backup',
                       help='Backup directory (default: data/output_backup)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("RE-PROCESSING VIDEOS WITH UPDATED FEATURES")
    logger.info("="*80)
    logger.info("New features added:")
    logger.info("  - skin_pixel_ratio (HSV-based skin detection)")
    logger.info("  - teacher_presence (black_pixel_ratio + skin_pixel_ratio)")
    logger.info("="*80)
    
    # Backup existing files unless --no-backup flag is set
    if not args.no_backup:
        backup_path = backup_existing_csvs(args.output, args.backup_dir)
        if backup_path:
            logger.info(f"\nBackup location: {backup_path}")
            logger.info("You can restore from backup if needed.")
    else:
        logger.warning("Skipping backup (--no-backup flag set)")
    
    # Re-process all videos
    logger.info(f"\nStarting video re-processing from: {args.input}")
    logger.info("="*80)
    
    process_video_folder(
        input_folder=args.input,
        output_folder=args.output,
        target_fps=args.fps
    )
    
    logger.info("="*80)
    logger.info("RE-PROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info("New CSV files now contain 25 features (was 23)")
    logger.info("New features: skin_pixel_ratio, teacher_presence")
    logger.info("="*80)


if __name__ == "__main__":
    main()
