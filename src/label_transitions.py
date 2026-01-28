"""
File: label_transitions.py
Purpose: Automatically label CSV files with ground truth transitions
Author: Smart Notes Generator Team
Created: 2026-01-26
Last Modified: 2026-01-26

This script:
1. Loads manual timestamps from ground_truth/[video_name]/transitions.txt
2. For each timestamp, finds the frame with lowest SSIM in a 3-second window (+/- 1.5s)
3. Labels that frame + 5 frames before/after as 1 (transition)
"""

import pandas as pd
from pathlib import Path
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/labeling.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TransitionLabeler:
    """
    Label CSV files with ground truth transitions based on manual timestamps.
    """
    
    def __init__(self, csv_dir: str = "data/output", ground_truth_dir: str = "data/ground_truth",
                 window_seconds: float = 1.5, context_frames: int = 5):
        """
        Initialize the transition labeler.
        
        Args:
            csv_dir: Directory containing feature CSV files
            ground_truth_dir: Directory containing ground truth timestamps
            window_seconds: Window size (+/-) in seconds to search for lowest SSIM (default: 1.5s)
            context_frames: Number of frames before/after to label (default: 5)
        """
        self.csv_dir = Path(csv_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.window_seconds = window_seconds
        self.context_frames = context_frames
        
        logger.info(f"Initialized TransitionLabeler")
        logger.info(f"  CSV Directory: {self.csv_dir}")
        logger.info(f"  Ground Truth Directory: {self.ground_truth_dir}")
        logger.info(f"  Window: +/- {window_seconds}s")
        logger.info(f"  Context: {context_frames} frames before/after")
    
    def load_timestamps(self, video_name: str) -> list:
        """
        Load manual timestamps for a video from ground_truth folder.
        
        Args:
            video_name: Name of the video (without _features.csv)
            
        Returns:
            List of timestamps in seconds
        """
        # Try to find the ground truth file
        gt_file = self.ground_truth_dir / video_name / "transitions.txt"
        
        if not gt_file.exists():
            logger.warning(f"Ground truth file not found: {gt_file}")
            return []
        
        timestamps = []
        
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        timestamp = float(line)
                        timestamps.append(timestamp)
                    except ValueError:
                        logger.warning(f"Could not parse timestamp: {line}")
            
            logger.info(f"Loaded {len(timestamps)} timestamps for {video_name}")
            return timestamps
            
        except Exception as e:
            logger.error(f"Error reading {gt_file}: {e}")
            return []
    
    def find_transition_frame(self, df: pd.DataFrame, timestamp: float) -> int:
        """
        Find the frame with lowest SSIM within the window around the timestamp.
        
        Args:
            df: DataFrame with video features
            timestamp: Manual timestamp in seconds
            
        Returns:
            Frame index with lowest SSIM, or -1 if not found
        """
        # Calculate window bounds
        window_start = timestamp - self.window_seconds
        window_end = timestamp + self.window_seconds
        
        # Filter frames within the window
        window_df = df[
            (df['timestamp_seconds'] >= window_start) &
            (df['timestamp_seconds'] <= window_end)
        ].copy()
        
        if len(window_df) == 0:
            logger.warning(f"No frames found in window [{window_start:.2f}, {window_end:.2f}]")
            return -1
        
        # Find frame with lowest global_ssim
        min_ssim_idx = window_df['global_ssim'].idxmin()
        min_ssim_frame = window_df.loc[min_ssim_idx, 'frame_index']
        min_ssim_value = window_df.loc[min_ssim_idx, 'global_ssim']
        min_ssim_timestamp = window_df.loc[min_ssim_idx, 'timestamp_seconds']
        
        logger.debug(f"  Manual: {timestamp:.2f}s → Found: {min_ssim_timestamp:.2f}s "
                    f"(frame {min_ssim_frame}, SSIM={min_ssim_value:.4f})")
        
        return int(min_ssim_frame)
    
    def label_frames(self, df: pd.DataFrame, frame_indices: list) -> pd.DataFrame:
        """
        Label frames and their context as transitions.
        
        Args:
            df: DataFrame with video features
            frame_indices: List of frame indices to label
            
        Returns:
            DataFrame with updated labels
        """
        # Reset all labels to 0
        df['label'] = 0
        
        labeled_count = 0
        
        for frame_idx in frame_indices:
            if frame_idx == -1:
                continue
            
            # Calculate range (frame +/- context_frames)
            start_frame = max(0, frame_idx - self.context_frames)
            end_frame = min(len(df) - 1, frame_idx + self.context_frames)
            
            # Label the range
            df.loc[
                (df['frame_index'] >= start_frame) &
                (df['frame_index'] <= end_frame),
                'label'
            ] = 1
            
            labeled_count += (end_frame - start_frame + 1)
        
        return df, labeled_count
    
    def process_video(self, csv_file: Path) -> bool:
        """
        Process a single video CSV file and label transitions.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract video name from filename
            video_name = csv_file.stem.replace('_features', '')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {video_name}")
            logger.info(f"{'='*60}")
            
            # Load timestamps
            timestamps = self.load_timestamps(video_name)
            
            if not timestamps:
                logger.warning(f"No timestamps found for {video_name}, skipping...")
                return False
            
            # Load CSV
            logger.info(f"Loading CSV: {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            logger.info(f"CSV loaded: {len(df)} frames, {len(df.columns)} columns")
            
            # Find transition frames
            transition_frames = []
            
            logger.info(f"Finding transitions for {len(timestamps)} manual timestamps...")
            for i, timestamp in enumerate(timestamps, 1):
                logger.info(f"  [{i}/{len(timestamps)}] Timestamp: {timestamp:.2f}s")
                frame_idx = self.find_transition_frame(df, timestamp)
                
                if frame_idx != -1:
                    transition_frames.append(frame_idx)
                    actual_timestamp = df[df['frame_index'] == frame_idx]['timestamp_seconds'].values[0]
                    logger.info(f"    -> Frame {frame_idx} @ {actual_timestamp:.2f}s")
            
            logger.info(f"Found {len(transition_frames)} valid transitions")
            
            # Label frames
            logger.info(f"Labeling frames (context: +/- {self.context_frames} frames)...")
            df, labeled_count = self.label_frames(df, transition_frames)
            
            # Save labeled CSV
            output_file = csv_file  # Overwrite original
            df.to_csv(output_file, index=False)
            
            total_transitions = df['label'].sum()
            logger.info(f"Labeled {labeled_count} frames ({total_transitions} total 1s)")
            logger.info(f"Saved to: {output_file}")
            logger.info(f"[SUCCESS] {video_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] Error processing {csv_file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_all(self) -> dict:
        """
        Process all CSV files in the directory.
        
        Returns:
            Dictionary with processing statistics
        """
        # Find all CSV files
        csv_files = list(self.csv_dir.glob("*_features.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found in {self.csv_dir}")
            return {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRANSITION LABELING - BATCH PROCESSING")
        logger.info(f"{'='*60}")
        logger.info(f"Found {len(csv_files)} CSV files to process")
        logger.info(f"{'='*60}\n")
        
        # Process each file
        stats = {
            'total': len(csv_files),
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for i, csv_file in enumerate(csv_files, 1):
            logger.info(f"\n[{i}/{len(csv_files)}] Processing: {csv_file.name}")
            
            success = self.process_video(csv_file)
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"LABELING COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Total videos: {stats['total']}")
        logger.info(f"Successfully labeled: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Success rate: {stats['success']/stats['total']*100:.1f}%")
        logger.info(f"{'='*60}\n")
        
        return stats


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Label CSV files with ground truth transitions'
    )
    parser.add_argument('--csv-dir', type=str, default='data/output',
                       help='Directory containing feature CSV files (default: data/output)')
    parser.add_argument('--ground-truth-dir', type=str, default='data/ground_truth',
                       help='Directory containing ground truth timestamps (default: data/ground_truth)')
    parser.add_argument('--window', type=float, default=1.5,
                       help='Window size (+/-) in seconds (default: 1.5)')
    parser.add_argument('--context', type=int, default=5,
                       help='Number of frames before/after to label (default: 5)')
    parser.add_argument('--single', type=str, default=None,
                       help='Process a single CSV file instead of all')
    
    args = parser.parse_args()
    
    # Create labeler
    labeler = TransitionLabeler(
        csv_dir=args.csv_dir,
        ground_truth_dir=args.ground_truth_dir,
        window_seconds=args.window,
        context_frames=args.context
    )
    
    # Process single file or all files
    if args.single:
        csv_file = Path(args.single)
        if not csv_file.exists():
            csv_file = Path(args.csv_dir) / args.single
        
        if csv_file.exists():
            labeler.process_video(csv_file)
        else:
            logger.error(f"File not found: {args.single}")
    else:
        labeler.process_all()


if __name__ == "__main__":
    main()
