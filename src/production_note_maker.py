"""
Production-Ready Lecture Note Maker
====================================
Final pipeline for extracting high-quality lecture notes from videos.

Features:
- Uses refined_predictions.csv (5 FPS) for transition detection
- Extracts frames at native 30 FPS for maximum quality
- Intelligent frame selection with teacher presence scoring
- Automatic deduplication with SSIM
- JSON metadata export for multimodal AI processing
- FFmpeg audio extraction commands

Author: Smart Notes Generator Team
Date: January 26, 2026
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import logging
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import json
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_notes_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionNoteMaker:
    """Production-ready lecture note extraction pipeline"""
    
    def __init__(self, video_dir='data/videos', predictions_csv='data/refined_predictions.csv',
                 output_dir='data/final_notes', lookback_seconds=5.0,
                 similarity_threshold=0.93, native_fps=30, min_gap_seconds=5.0):
        """
        Initialize the production note maker
        
        Args:
            video_dir: Directory containing video files
            predictions_csv: Path to refined_predictions.csv
            output_dir: Directory to save extracted notes
            lookback_seconds: Time window to look back from transition (default 5s)
            similarity_threshold: SSIM threshold for deduplication (default 0.93)
            native_fps: Native video frame rate (default 30 FPS)
            min_gap_seconds: Minimum time gap between transitions (default 5s)
        """
        self.video_dir = Path(video_dir)
        self.predictions_csv = Path(predictions_csv)
        self.output_dir = Path(output_dir)
        self.lookback_seconds = lookback_seconds
        self.similarity_threshold = similarity_threshold
        self.native_fps = native_fps
        self.min_gap_seconds = min_gap_seconds
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Initialized ProductionNoteMaker")
        logger.info(f"  Video Directory: {self.video_dir}")
        logger.info(f"  Predictions CSV: {self.predictions_csv}")
        logger.info(f"  Output Directory: {self.output_dir}")
        logger.info(f"  Lookback Window: {self.lookback_seconds} seconds")
        logger.info(f"  Similarity Threshold: {self.similarity_threshold}")
        logger.info(f"  Native FPS: {self.native_fps}")
        logger.info(f"  Minimum Gap: {self.min_gap_seconds} seconds")
    
    def find_video_file(self, video_id):
        """Find video file for given video_id"""
        extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
        
        for ext in extensions:
            video_path = self.video_dir / f"{video_id}{ext}"
            if video_path.exists():
                return video_path
        
        return None
    
    def calculate_teacher_presence_realtime(self, frame):
        """
        Calculate teacher presence for a frame in real-time
        Uses black pixel ratio + skin pixel ratio (same as training)
        
        Args:
            frame: BGR image from video
            
        Returns:
            float: teacher_presence score (0 to 1)
        """
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin detection (HSV range)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixel_ratio = np.count_nonzero(skin_mask) / (frame.shape[0] * frame.shape[1])
        
        # Black pixel detection (RGB < 50)
        black_mask = np.all(frame < 50, axis=2)
        black_pixel_ratio = np.count_nonzero(black_mask) / (frame.shape[0] * frame.shape[1])
        
        # Teacher presence = black pixels + skin pixels
        teacher_presence = np.clip(black_pixel_ratio + skin_pixel_ratio, 0, 1)
        
        return teacher_presence
    
    def extract_high_res_frames_from_window(self, video_path, transition_timestamp):
        """
        Extract frames at native 30 FPS from lookback window
        Calculate quality scores in real-time
        
        Args:
            video_path: Path to video file
            transition_timestamp: Timestamp of transition (seconds)
            
        Returns:
            List of dicts with frame info and scores
        """
        window_start = max(0, transition_timestamp - self.lookback_seconds)
        window_end = transition_timestamp
        
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Convert timestamps to frame numbers at native FPS
        start_frame = int(window_start * video_fps)
        end_frame = int(window_end * video_fps)
        
        frames_data = []
        
        # Extract all frames in window
        for frame_num in range(start_frame, end_frame + 1):
            # Set position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calculate timestamp
            timestamp = frame_num / video_fps
            
            # Calculate teacher presence in real-time
            teacher_presence = self.calculate_teacher_presence_realtime(frame)
            
            # Calculate progress bias (0 to 1, higher = closer to transition)
            progress = (frame_num - start_frame) / max(end_frame - start_frame, 1)
            
            # Calculate quality score
            # Score = (1 - teacher_presence) * (1 + progress)
            quality_score = (1 - teacher_presence) * (1 + progress)
            
            frames_data.append({
                'frame_num': frame_num,
                'timestamp': timestamp,
                'frame': frame,
                'teacher_presence': teacher_presence,
                'progress': progress,
                'quality_score': quality_score
            })
        
        cap.release()
        
        logger.info(f"    Extracted {len(frames_data)} frames at {video_fps} FPS")
        
        return frames_data
    
    def calculate_frame_similarity(self, img1, img2):
        """Calculate SSIM between two frames"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        if gray1.shape != gray2.shape:
            h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))
        
        similarity = ssim(gray1, gray2)
        return similarity
    
    def process_video(self, video_id):
        """
        Process a single video and extract notes
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of note metadata dictionaries
        """
        logger.info("\n" + "="*70)
        logger.info(f"Processing: {video_id}")
        logger.info("="*70)
        
        # Find video file
        video_path = self.find_video_file(video_id)
        if video_path is None:
            logger.error(f"Video file not found for {video_id}")
            return []
        
        logger.info(f"Video: {video_path}")
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps
        cap.release()
        
        logger.info(f"Duration: {duration:.2f}s, FPS: {video_fps}, Total Frames: {total_frames}")
        
        # Load predictions
        predictions_df = pd.read_csv(self.predictions_csv)
        video_preds = predictions_df[predictions_df['video_id'] == video_id].copy()
        
        if len(video_preds) == 0:
            logger.warning(f"No predictions found for {video_id}")
            return []
        
        # Find transitions (smoothed_prediction == 1)
        transitions = video_preds[video_preds['smoothed_prediction'] == 1].copy()
        transitions = transitions.sort_values('timestamp_seconds')
        
        logger.info(f"Found {len(transitions)} transition events")
        
        # Add final transition at end of video (ensure last slide is captured)
        final_transition = {
            'video_id': video_id,
            'timestamp_seconds': duration,
            'smoothed_prediction': 1
        }
        transitions = pd.concat([transitions, pd.DataFrame([final_transition])], ignore_index=True)
        
        logger.info(f"Found {len(transitions)-1} raw transition events")
        
        # Filter transitions that are too close together (minimum gap)
        transition_times = sorted(transitions['timestamp_seconds'].unique())
        filtered_times = [transition_times[0]]
        for t in transition_times[1:]:
            if t - filtered_times[-1] >= self.min_gap_seconds:
                filtered_times.append(t)
        
        logger.info(f"Filtered: {len(transition_times)} -> {len(filtered_times)} transitions (min gap: {self.min_gap_seconds}s)")
        logger.info(f"Skipped {len(transition_times) - len(filtered_times)} transitions too close to previous ones")
        
        # Keep only filtered transitions
        transitions = transitions[transitions['timestamp_seconds'].isin(filtered_times)].copy()
        logger.info(f"Total transitions to process: {len(transitions)}")
        
        # Create output directory for this video
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(exist_ok=True)
        
        notes_metadata = []
        previous_frame = None
        previous_note_id = None
        
        # Process each transition
        for idx, transition in transitions.iterrows():
            transition_time = transition['timestamp_seconds']
            note_num = len(notes_metadata) + 1
            
            logger.info(f"\n[{note_num}/{len(transitions)}] Transition at {transition_time:.2f}s")
            
            # Extract high-res frames from window
            logger.info(f"  Extracting frames at {self.native_fps} FPS...")
            frames_data = self.extract_high_res_frames_from_window(video_path, transition_time)
            
            if len(frames_data) == 0:
                logger.warning("  No frames extracted from window")
                continue
            
            # Find frame with highest quality score
            best_frame_data = max(frames_data, key=lambda x: x['quality_score'])
            
            logger.info(f"  Best frame: {best_frame_data['timestamp']:.2f}s")
            logger.info(f"    Teacher presence: {best_frame_data['teacher_presence']:.3f}")
            logger.info(f"    Quality score: {best_frame_data['quality_score']:.3f}")
            
            # Check for duplicates with previous frame
            if previous_frame is not None:
                similarity = self.calculate_frame_similarity(
                    best_frame_data['frame'], previous_frame
                )
                
                logger.info(f"  Similarity to previous: {similarity:.4f}")
                
                if similarity >= self.similarity_threshold:
                    # Duplicate detected
                    prev_teacher = notes_metadata[-1]['teacher_presence_score']
                    curr_teacher = best_frame_data['teacher_presence']
                    
                    logger.info(f"  DUPLICATE detected (SSIM >= {self.similarity_threshold})")
                    logger.info(f"    Previous teacher: {prev_teacher:.3f}")
                    logger.info(f"    Current teacher:  {curr_teacher:.3f}")
                    
                    if curr_teacher < prev_teacher:
                        # Current is cleaner - replace previous
                        logger.info(f"  -> Replacing {previous_note_id} with cleaner frame")
                        
                        # Delete previous image
                        prev_image_path = video_output_dir / f"{previous_note_id}.png"
                        if prev_image_path.exists():
                            prev_image_path.unlink()
                        
                        # Update metadata
                        notes_metadata[-1]['capture_timestamp'] = best_frame_data['timestamp']
                        notes_metadata[-1]['teacher_presence_score'] = curr_teacher
                        notes_metadata[-1]['quality_score'] = best_frame_data['quality_score']
                        
                        # Save new image
                        cv2.imwrite(str(prev_image_path), best_frame_data['frame'])
                        previous_frame = best_frame_data['frame']
                        
                    else:
                        # Previous is cleaner - skip current
                        logger.info(f"  -> Keeping {previous_note_id} (already cleaner)")
                        continue
                else:
                    # Not a duplicate - save as new note
                    note_id = f"{video_id}_note_{note_num:03d}"
                    image_path = video_output_dir / f"{note_id}.png"
                    
                    # Save frame as high-quality PNG
                    cv2.imwrite(str(image_path), best_frame_data['frame'])
                    logger.info(f"  Saved: {image_path.name}")
                    
                    # Determine audio segment boundaries
                    if len(notes_metadata) > 0:
                        audio_start = notes_metadata[-1]['audio_segment_end']
                    else:
                        audio_start = 0.0
                    
                    audio_end = transition_time
                    
                    # Store metadata
                    notes_metadata.append({
                        'slide_id': note_id,
                        'capture_timestamp': best_frame_data['timestamp'],
                        'audio_segment_start': audio_start,
                        'audio_segment_end': audio_end,
                        'teacher_presence_score': best_frame_data['teacher_presence'],
                        'quality_score': best_frame_data['quality_score'],
                        'image_path': str(image_path)
                    })
                    
                    previous_frame = best_frame_data['frame']
                    previous_note_id = note_id
            else:
                # First frame - no comparison needed
                note_id = f"{video_id}_note_{note_num:03d}"
                image_path = video_output_dir / f"{note_id}.png"
                
                cv2.imwrite(str(image_path), best_frame_data['frame'])
                logger.info(f"  Saved: {image_path.name}")
                
                notes_metadata.append({
                    'slide_id': note_id,
                    'capture_timestamp': best_frame_data['timestamp'],
                    'audio_segment_start': 0.0,
                    'audio_segment_end': transition_time,
                    'teacher_presence_score': best_frame_data['teacher_presence'],
                    'quality_score': best_frame_data['quality_score'],
                    'image_path': str(image_path)
                })
                
                previous_frame = best_frame_data['frame']
                previous_note_id = note_id
        
        logger.info(f"\n{video_id} Complete:")
        logger.info(f"  Extracted {len(notes_metadata)} unique high-quality notes")
        
        # Generate metadata and FFmpeg scripts for this video
        if notes_metadata:
            self.generate_metadata_json(video_id, notes_metadata)
            self.generate_ffmpeg_commands(video_id, notes_metadata)
        
        return notes_metadata
    
    def generate_metadata_json(self, video_id, notes_metadata):
        """
        Generate metadata.json file for a specific video
        
        Args:
            video_id: Video identifier
            notes_metadata: List of note metadata dictionaries
        """
        logger.info("\nGenerating Metadata JSON")
        logger.info("="*50)
        
        # Create metadata structure
        metadata = {
            'video_id': video_id,
            'generated_at': datetime.now().isoformat(),
            'total_slides': len(notes_metadata),
            'lookback_seconds': self.lookback_seconds,
            'min_gap_seconds': self.min_gap_seconds,
            'slides': []
        }
        
        for note in notes_metadata:
            metadata['slides'].append({
                'slide_id': note['slide_id'],
                'capture_timestamp': round(note['capture_timestamp'], 2),
                'audio_segment_start': round(note['audio_segment_start'], 2),
                'audio_segment_end': round(note['audio_segment_end'], 2),
                'teacher_presence_score': round(note['teacher_presence_score'], 3),
                'quality_score': round(note['quality_score'], 3)
            })
        
        # Save JSON in video-specific folder
        video_output_dir = self.output_dir / video_id
        metadata_path = video_output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved: {metadata_path}")
        logger.info(f"Total slides: {len(notes_metadata)}")
        
        return metadata
    
    def generate_ffmpeg_commands(self, video_id, notes_metadata):
        """
        Generate FFmpeg commands for audio extraction for a specific video
        
        Args:
            video_id: Video identifier
            notes_metadata: List of note metadata dictionaries
        """
        logger.info("\nGenerating FFmpeg Audio Extraction Commands")
        logger.info("="*50)
        
        video_output_dir = self.output_dir / video_id
        video_path = self.video_dir / f"{video_id}.mp4"
        audio_dir = video_output_dir / 'audio'
        
        # Create bash script
        ffmpeg_script_path = video_output_dir / 'extract_audio.sh'
        
        with open(ffmpeg_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Audio extraction for {video_id}\n")
            f.write("# Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            
            f.write(f"mkdir -p {audio_dir}\n\n")
            
            for note in notes_metadata:
                slide_id = note['slide_id']
                start = note['audio_segment_start']
                end = note['audio_segment_end']
                duration = end - start
                
                output_audio = f"{audio_dir}/{slide_id}.mp3"
                
                # FFmpeg command
                cmd = (f"ffmpeg -i \\\"{video_path}\\\" "
                      f"-ss {start:.2f} -t {duration:.2f} "
                      f"-vn -acodec libmp3lame -q:a 2 "
                      f"\\\"{output_audio}\\\"\\n")
                
                f.write(cmd)
        
        logger.info(f"Saved: {ffmpeg_script_path}")
        
        # Also create Windows batch script
        batch_script_path = video_output_dir / 'extract_audio.bat'
        
        with open(batch_script_path, 'w') as f:
            f.write("@echo off\n")
            f.write(f"REM Audio extraction for {video_id}\n")
            f.write("REM Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            
            audio_dir_win = str(audio_dir).replace('/', '\\\\')
            f.write(f"if not exist \\\"{audio_dir_win}\\\" mkdir \\\"{audio_dir_win}\\\"\\n\\n")
            
            for note in notes_metadata:
                slide_id = note['slide_id']
                start = note['audio_segment_start']
                end = note['audio_segment_end']
                duration = end - start
                
                output_audio = f"{audio_dir_win}\\\\{slide_id}.mp3"
                
                cmd = (f"ffmpeg -i \\\"{video_path}\\\" "
                      f"-ss {start:.2f} -t {duration:.2f} "
                      f"-vn -acodec libmp3lame -q:a 2 "
                      f"\\\"{output_audio}\\\"\\n")
                
                f.write(cmd)
        
        logger.info(f"Saved: {batch_script_path}")
        logger.info(f"\\nTo extract audio:")
        logger.info(f"  Windows: {batch_script_path}")
        logger.info(f"  Linux/Mac: bash {ffmpeg_script_path}")
    
    def run_pipeline(self, single_video=None):
        """
        Execute complete production pipeline
        
        Args:
            single_video: Optional video_id to process only one video
        """
        logger.info("\n" + "="*70)
        logger.info("PRODUCTION NOTE MAKER - FINAL PIPELINE")
        logger.info("="*70)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if single_video:
            logger.info(f"MODE: Single Video - {single_video}")
        else:
            logger.info("MODE: All Videos")
        
        logger.info("\nPipeline Steps:")
        logger.info("1. Load refined predictions (5 FPS)")
        logger.info("2. Extract frames at native 30 FPS with quality scoring")
        logger.info("3. Intelligent deduplication (SSIM > 0.93)")
        logger.info("4. Export metadata JSON")
        logger.info("5. Generate FFmpeg audio extraction commands")
        
        try:
            # Load predictions
            logger.info("\n" + "="*70)
            logger.info("Loading Predictions")
            logger.info("="*70)
            
            if not self.predictions_csv.exists():
                logger.error(f"Predictions CSV not found: {self.predictions_csv}")
                return False
            
            predictions_df = pd.read_csv(self.predictions_csv)
            logger.info(f"Loaded: {self.predictions_csv}")
            logger.info(f"  Total frames: {len(predictions_df)}")
            logger.info(f"  Videos: {predictions_df['video_id'].nunique()}")
            
            # Get unique videos
            if single_video:
                # Process only the specified video
                if single_video not in predictions_df['video_id'].values:
                    logger.error(f"Video '{single_video}' not found in predictions")
                    logger.info(f"Available videos: {', '.join(sorted(predictions_df['video_id'].unique()))}")
                    return False
                video_ids = [single_video]
                logger.info(f"\nProcessing 1 video: {single_video}")
            else:
                # Process all videos
                video_ids = predictions_df['video_id'].unique()
                logger.info(f"\nProcessing {len(video_ids)} videos")
            
            all_notes = {}
            
            # Process each video
            for video_id in sorted(video_ids):
                notes_metadata = self.process_video(video_id)
                all_notes[video_id] = notes_metadata
            
            if len(all_notes) == 0:
                logger.warning("No notes extracted!")
                return False
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETE!")
            logger.info("="*70)
            total_slides = sum(len(notes) for notes in all_notes.values())
            logger.info(f"\nResults:")
            logger.info(f"  Total videos: {len(video_ids)}")
            logger.info(f"  Total slides: {total_slides}")
            logger.info(f"  Output directory: {self.output_dir}")
            logger.info(f"\nEach video folder contains:")
            logger.info(f"  - High-quality PNG slides")
            logger.info(f"  - metadata.json (timestamps, quality scores)")
            logger.info(f"  - extract_audio.bat (Windows FFmpeg script)")
            logger.info(f"  - extract_audio.sh (Linux/Mac FFmpeg script)")
            logger.info("\nNext Steps:")
            logger.info("1. Review extracted slides in data/final_notes/<video_id>/")
            logger.info("2. Extract audio: Run extract_audio.bat in each video folder")
            logger.info("3. Feed slides + metadata.json + audio to multimodal AI")
            logger.info("2. Run audio extraction: extract_audio.bat")
            logger.info("3. Use lecture_metadata.json for multimodal AI processing")
            logger.info("3. Feed slides + metadata.json + audio to multimodal AI")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Production-Ready Lecture Note Maker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python src/production_note_maker.py --video algo_1
  
  # Process all videos
  python src/production_note_maker.py
  
  # Use custom predictions file
  python src/production_note_maker.py --predictions data/my_predictions.csv --video toc_1
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Process only this specific video (e.g., algo_1, toc_1)'
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        default='data/all_predictions.csv',
        help='Path to predictions CSV (default: data/all_predictions.csv)'
    )
    
    parser.add_argument(
        '--list-videos',
        action='store_true',
        help='List all available videos and exit'
    )
    
    args = parser.parse_args()
    
    # List videos if requested
    if args.list_videos:
        predictions_path = Path(args.predictions)
        if predictions_path.exists():
            df = pd.read_csv(predictions_path)
            videos = sorted(df['video_id'].unique())
            print(f"\nAvailable videos in {args.predictions}:")
            print("-" * 50)
            for i, vid in enumerate(videos, 1):
                transitions = (df[df['video_id']==vid]['smoothed_prediction']==1).sum()
                print(f"  {i:2d}. {vid:30s} ({transitions} transitions)")
            print(f"\nTotal: {len(videos)} videos")
        else:
            print(f"Error: Predictions file not found: {args.predictions}")
        return
    
    # Run pipeline
    note_maker = ProductionNoteMaker(predictions_csv=args.predictions)
    success = note_maker.run_pipeline(single_video=args.video)
    
    if success:
        logger.info("\n SUCCESS - Production notes ready!")
    else:
        logger.error("\n FAILED - Check logs for details")


if __name__ == '__main__':
    main()
