"""
Final Slide Generator - Non-Redundant Transition Detection
Implements sequential comparison with SSIM-based deduplication and blank slide detection
"""

import cv2
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalSlideGenerator:
    """Generate non-redundant final slide list with SSIM-based deduplication"""
    
    def __init__(self, video_dir='data/videos', predictions_csv='data/all_predictions.csv',
                 features_dir='data/output', lectures_dir='data/lectures',
                 prediction_threshold=0.10, ssim_dedup_threshold=0.95,
                 ssim_rapid_threshold=0.85, rapid_fire_window=2.0,
                 blank_edge_threshold=0.1):
        
        self.video_dir = Path(video_dir)
        self.predictions_csv = Path(predictions_csv)
        self.features_dir = Path(features_dir)
        self.lectures_dir = Path(lectures_dir)
        
        # Thresholds
        self.prediction_threshold = prediction_threshold
        self.ssim_dedup_threshold = ssim_dedup_threshold
        self.ssim_rapid_threshold = ssim_rapid_threshold
        self.rapid_fire_window = rapid_fire_window
        self.blank_edge_threshold = blank_edge_threshold
        
        # Create lectures directory
        self.lectures_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Initialized FinalSlideGenerator")
        logger.info(f"  Prediction Threshold: {self.prediction_threshold}")
        logger.info(f"  SSIM Dedup Threshold: {self.ssim_dedup_threshold}")
        logger.info(f"  SSIM Rapid-Fire Threshold: {self.ssim_rapid_threshold}")
        logger.info(f"  Rapid-Fire Window: {self.rapid_fire_window}s")
        logger.info(f"  Blank Edge Threshold: {self.blank_edge_threshold}")
    
    def load_transitions(self, video_id):
        """Load transitions.json for a video"""
        transitions_file = self.lectures_dir / video_id / 'transitions.json'
        if not transitions_file.exists():
            logger.warning(f"Transitions not found: {transitions_file}")
            return None
        
        with open(transitions_file, 'r') as f:
            transitions = json.load(f)
        return transitions
    
    def load_features(self, video_id):
        """Load feature CSV for a video"""
        feature_file = self.features_dir / f"{video_id}_features.csv"
        if not feature_file.exists():
            logger.warning(f"Features not found: {feature_file}")
            return None
        
        features_df = pd.read_csv(feature_file)
        return features_df
    
    def extract_frame_at_timestamp(self, video_path, timestamp):
        """Extract a single frame at given timestamp"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        if ret:
            return frame
        return None
    
    def calculate_frame_ssim(self, frame1, frame2):
        """
        Calculate SSIM between two frames focusing on BOARD CONTENT ONLY.
        Applies mask to ignore teacher/person regions for better duplicate detection.
        """
        if frame1 is None or frame2 is None:
            return 0.0
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize to same dimensions if needed
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Create mask to focus on board area (ignore teacher regions)
        # Detect dark/bright pixels (board + chalk/marker) vs skin tones
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        
        # Board mask: dark backgrounds (blackboard/greenboard) or bright areas (whiteboard/chalk)
        # Exclude skin tones to ignore teacher movement
        lower_skin = np.array([0, 20, 0], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv1, lower_skin, upper_skin)
        
        # Invert to get board-only mask (1 = board, 0 = teacher/skin)
        board_mask = cv2.bitwise_not(skin_mask)
        
        # Apply mask to grayscale images
        gray1_masked = cv2.bitwise_and(gray1, gray1, mask=board_mask)
        gray2_masked = cv2.bitwise_and(gray2, gray2, mask=board_mask)
        
        # Calculate SSIM on masked (board-only) regions
        # This ignores teacher movement and focuses on actual board content
        score = ssim(gray1_masked, gray2_masked)
        return score
    
    def is_blank_slide(self, edge_count_normalized):
        """Determine if a slide is blank based on edge count"""
        return edge_count_normalized < self.blank_edge_threshold
    
    def group_consecutive_predictions(self, predictions_df):
        """
        Group consecutive predictions into bursts
        
        Returns:
            List of burst dictionaries with start_time, end_time, timestamps
        """
        timestamps = sorted(predictions_df['timestamp_seconds'].unique())
        
        if len(timestamps) == 0:
            return []
        
        bursts = []
        current_burst = {
            'start_time': timestamps[0],
            'end_time': timestamps[0],
            'timestamps': [timestamps[0]]
        }
        
        for i in range(1, len(timestamps)):
            time_gap = timestamps[i] - timestamps[i-1]
            
            if time_gap <= 0.5:  # Within 0.5s = same burst
                current_burst['end_time'] = timestamps[i]
                current_burst['timestamps'].append(timestamps[i])
            else:
                # Save current burst and start new one
                bursts.append(current_burst)
                current_burst = {
                    'start_time': timestamps[i],
                    'end_time': timestamps[i],
                    'timestamps': [timestamps[i]]
                }
        
        # Add final burst
        bursts.append(current_burst)
        
        return bursts
    
    def select_best_from_burst(self, burst, features_df):
        """
        Select the frame with lowest SSIM from a burst
        
        Args:
            burst: Dictionary with timestamps
            features_df: DataFrame with features including global_ssim
            
        Returns:
            Best timestamp (float)
        """
        # Get features for all timestamps in burst
        burst_features = features_df[
            features_df['timestamp_seconds'].isin(burst['timestamps'])
        ].copy()
        
        if len(burst_features) == 0:
            # Fallback to middle of burst
            return burst['timestamps'][len(burst['timestamps']) // 2]
        
        # Find frame with lowest SSIM (highest change)
        best_row = burst_features.loc[burst_features['global_ssim'].idxmin()]
        return best_row['timestamp_seconds']
    
    def process_video(self, video_id, use_predictions=False):
        """
        Process a single video to generate final unique slides
        
        Args:
            video_id: Video identifier
            use_predictions: If True, use predictions.csv instead of transitions.json
            
        Returns:
            List of final slide dictionaries
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {video_id}")
        logger.info(f"{'='*70}")
        
        # Create slides output directory
        slides_dir = self.lectures_dir / video_id / 'slides'
        slides_dir.mkdir(exist_ok=True, parents=True)
        
        # Load features
        features_df = self.load_features(video_id)
        if features_df is None:
            logger.error(f"Cannot process without features for {video_id}")
            return []
        
        # Find video file
        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return []
        
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        logger.info(f"Video duration: {video_duration:.2f}s")
        
        # Choose source based on flag
        if use_predictions:
            # Load predictions CSV from lectures directory
            predictions_file = self.lectures_dir / video_id / f"{video_id}_predictions.csv"
            if not predictions_file.exists():
                logger.error(f"Predictions not found: {predictions_file}")
                return []
            
            predictions_df = pd.read_csv(predictions_file)
            
            # Filter by prediction threshold
            positive_predictions = predictions_df[
                predictions_df['smoothed_prediction'] >= self.prediction_threshold
            ].copy()
            
            logger.info(f"Loaded {len(predictions_df)} predictions")
            logger.info(f"Filtered to {len(positive_predictions)} transitions (threshold={self.prediction_threshold})")
            
            # Group consecutive predictions into bursts
            bursts = self.group_consecutive_predictions(positive_predictions)
            logger.info(f"Grouped into {len(bursts)} bursts")
            
            # Select best frame from each burst
            candidate_timestamps = []
            for burst in bursts:
                best_ts = self.select_best_from_burst(burst, features_df)
                candidate_timestamps.append(best_ts)
            
            logger.info(f"Selected {len(candidate_timestamps)} candidate transitions from predictions")
        else:
            # Use deduplicated transitions from transitions.json
            transitions = self.load_transitions(video_id)
            if transitions is None:
                logger.error(f"Cannot process without transitions for {video_id}")
                return []
            
            candidate_timestamps = [t['timestamp'] for t in transitions['transitions']]
            logger.info(f"Using {len(candidate_timestamps)} deduplicated transitions from transitions.json")
        
        # Step 4: Sequential comparison with SSIM deduplication
        final_slides = []
        previous_frame = None
        previous_timestamp = 0.0
        
        for idx, current_ts in enumerate(candidate_timestamps):
            # Extract current frame
            current_frame = self.extract_frame_at_timestamp(video_path, current_ts)
            
            if current_frame is None:
                logger.warning(f"Could not extract frame at {current_ts:.2f}s")
                continue
            
            # Get edge count for blank detection
            current_features = features_df[
                (features_df['timestamp_seconds'] >= current_ts - 0.1) &
                (features_df['timestamp_seconds'] <= current_ts + 0.1)
            ]
            
            if len(current_features) > 0:
                edge_count = current_features.iloc[0]['edge_count']  # Already normalized
            else:
                edge_count = 0.5  # Default
            
            # Check if blank slide
            is_blank = self.is_blank_slide(edge_count)
            
            # First slide - always accept
            if previous_frame is None:
                ssim_from_prev = 0.0
                accept = True
            else:
                # Calculate SSIM with previous accepted slide
                ssim_from_prev = self.calculate_frame_ssim(current_frame, previous_frame)
                
                time_gap = current_ts - previous_timestamp
                
                # Rapid-fire logic: < 2s apart + different content
                if time_gap < self.rapid_fire_window and ssim_from_prev < self.ssim_rapid_threshold:
                    accept = True
                    logger.info(f"  [{idx+1}] RAPID-FIRE at {current_ts:.2f}s (gap={time_gap:.2f}s, SSIM={ssim_from_prev:.3f})")
                # Normal deduplication: SSIM check
                elif ssim_from_prev < self.ssim_dedup_threshold:
                    accept = True
                    logger.info(f"  [{idx+1}] ACCEPTED at {current_ts:.2f}s (SSIM={ssim_from_prev:.3f})")
                else:
                    accept = False
                    logger.info(f"  [{idx+1}] DUPLICATE at {current_ts:.2f}s (SSIM={ssim_from_prev:.3f}) - SKIPPED")
            
            if accept:
                # Determine audio window
                audio_start = previous_timestamp if len(final_slides) > 0 else 0.0
                audio_end = current_ts
                
                slide_number = len(final_slides) + 1
                
                # Save high-quality PNG to slides directory
                slide_filename = f"{video_id}_slide_{slide_number:03d}.png"
                slide_path = slides_dir / slide_filename
                cv2.imwrite(str(slide_path), current_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                
                slide_data = {
                    'slide_number': slide_number,
                    'filename': slide_filename,
                    'timestamp': round(current_ts, 2),
                    'is_blank': bool(is_blank),
                    'edge_count': round(float(edge_count), 3),
                    'ssim_from_prev': round(float(ssim_from_prev), 3),
                    'audio_window_start': round(audio_start, 2),
                    'audio_window_end': round(audio_end, 2),
                    'audio_duration': round(audio_end - audio_start, 2)
                }
                
                final_slides.append(slide_data)
                previous_frame = current_frame.copy()
                previous_timestamp = current_ts
                
                logger.info(f"    Saved: {slide_filename}")
                
                if is_blank:
                    logger.warning(f"    ⚠️ BLANK SLIDE detected (edge={edge_count:.3f})")
        
        # Save metadata.json for this video
        metadata = {
            'video_id': video_id,
            'extraction_method': 'ssim_deduplication',
            'extraction_params': {
                'ssim_dedup_threshold': self.ssim_dedup_threshold,
                'ssim_rapid_threshold': self.ssim_rapid_threshold,
                'blank_edge_threshold': self.blank_edge_threshold
            },
            'slides': final_slides
        }
        
        metadata_path = self.lectures_dir / video_id / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nSaved metadata: {metadata_path}")
        
        # Skip final capture at video end (not needed with deduplicated transitions)
        final_capture_time = video_duration - 1.0
        if False:  # Disabled
            logger.info(f"\nForcing final capture at {final_capture_time:.2f}s")
            
            final_frame = self.extract_frame_at_timestamp(video_path, final_capture_time)
            
            if final_frame is not None:
                # Get edge count
                final_features = features_df[
                    (features_df['timestamp_seconds'] >= final_capture_time - 0.5) &
                    (features_df['timestamp_seconds'] <= final_capture_time + 0.5)
                ]
                
                if len(final_features) > 0:
                    edge_count = final_features.iloc[0]['edge_count']  # Already normalized
                else:
                    edge_count = 0.5
                
                is_blank = self.is_blank_slide(edge_count)
                
                # Calculate SSIM with last accepted slide
                if previous_frame is not None:
                    ssim_from_prev = self.calculate_frame_ssim(final_frame, previous_frame)
                else:
                    ssim_from_prev = 0.0
                
                # Only add if different from last slide
                if ssim_from_prev < self.ssim_dedup_threshold:
                    audio_start = previous_timestamp if len(final_slides) > 0 else 0.0
                    
                    final_slide = {
                        'slide_number': len(final_slides) + 1,
                        'timestamp': round(final_capture_time, 2),
                        'is_blank': bool(is_blank),
                        'edge_count': round(float(edge_count), 3),
                        'ssim_from_prev': round(float(ssim_from_prev), 3),
                        'audio_window_start': round(audio_start, 2),
                        'audio_window_end': round(final_capture_time, 2),
                        'audio_duration': round(final_capture_time - audio_start, 2)
                    }
                    
                    final_slides.append(final_slide)
                    logger.info(f"  FINAL SLIDE added (SSIM={ssim_from_prev:.3f})")
                else:
                    logger.info(f"  Final capture skipped (SSIM={ssim_from_prev:.3f} too similar)")
        
        logger.info(f"\n✅ Final result: {len(final_slides)} unique slides")
        blank_count = sum(1 for s in final_slides if s['is_blank'])
        logger.info(f"   Blank slides: {blank_count}")
        logger.info(f"   Content slides: {len(final_slides) - blank_count}")
        
        return final_slides
    
    def process_all_videos(self):
        """Process all videos in lectures directory"""
        logger.info("\n" + "="*70)
        logger.info("FINAL SLIDE GENERATION - ALL VIDEOS")
        logger.info("="*70)
        
        # Get all video directories from lectures folder
        video_dirs = [d for d in self.lectures_dir.iterdir() if d.is_dir()]
        video_ids = sorted([d.name for d in video_dirs])
        
        logger.info(f"Found {len(video_ids)} videos to process")
        
        all_results = {}
        
        for idx, video_id in enumerate(video_ids, 1):
            logger.info(f"\n[{idx}/{len(video_ids)}]")
            
            final_slides = self.process_video(video_id)
            
            if final_slides:
                all_results[video_id] = final_slides
        
        logger.info("\n" + "="*70)
        logger.info("EXTRACTION COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total videos: {len(all_results)}")
        logger.info(f"Total slides extracted: {sum(len(slides) for slides in all_results.values())}")
        
        # Generate summary per video
        for video_id, slides in all_results.items():
            blank_count = sum(1 for s in slides if s['is_blank'])
            logger.info(f"  {video_id}: {len(slides)} slides ({blank_count} blank)")
        
        return all_results


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate final non-redundant slide list with SSIM deduplication',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, help='Process single video')
    parser.add_argument('--all', action='store_true', help='Process all videos')
    parser.add_argument('--use-predictions', action='store_true', 
                       help='Use predictions.csv instead of transitions.json')
    parser.add_argument('--ssim-dedup', type=float, default=0.95,
                       help='SSIM threshold for deduplication (default: 0.95, board-only comparison)')
    parser.add_argument('--ssim-rapid', type=float, default=0.85,
                       help='SSIM threshold for rapid-fire detection (default: 0.85)')
    parser.add_argument('--blank-threshold', type=float, default=0.1,
                       help='Edge count threshold for blank slides (default: 0.1)')
    
    args = parser.parse_args()
    
    generator = FinalSlideGenerator(
        ssim_dedup_threshold=args.ssim_dedup,
        ssim_rapid_threshold=args.ssim_rapid,
        blank_edge_threshold=args.blank_threshold
    )
    
    if args.video:
        # Single video
        slides = generator.process_video(args.video, use_predictions=args.use_predictions)
        logger.info(f"\n✅ Extracted {len(slides)} slides for {args.video}")
        
    elif args.all:
        # All videos
        generator.process_all_videos()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
