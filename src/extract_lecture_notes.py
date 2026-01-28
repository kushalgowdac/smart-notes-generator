"""
Lecture Notes Extractor
=======================
Extracts key frames from lecture videos at transition points,
using intelligent frame selection to minimize teacher presence.

Algorithm:
1. For each transition, look back 10 seconds (50 frames)
2. Calculate Selection_Score = (1 - teacher_presence) * (1 + frame_position_weight)
3. Extract frame with highest score (least teacher, closest to transition)
4. Deduplicate similar frames using SSIM

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
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/notes_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NotesExtractor:
    """Extract lecture notes frames from videos at transition points"""
    
    def __init__(self, video_dir='data/videos', csv_dir='data/output', 
                 output_dir='data/extracted_notes', lookback_seconds=10.0,
                 similarity_threshold=0.92):
        """
        Initialize the notes extractor
        
        Args:
            video_dir: Directory containing video files
            csv_dir: Directory containing feature CSVs
            output_dir: Directory to save extracted frames
            lookback_seconds: Time window to look back from transition (default 10s)
            similarity_threshold: SSIM threshold for deduplication (default 0.92)
        """
        self.video_dir = Path(video_dir)
        self.csv_dir = Path(csv_dir)
        self.output_dir = Path(output_dir)
        self.lookback_seconds = lookback_seconds
        self.similarity_threshold = similarity_threshold
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Initialized NotesExtractor")
        logger.info(f"  Video Directory: {self.video_dir}")
        logger.info(f"  CSV Directory: {self.csv_dir}")
        logger.info(f"  Output Directory: {self.output_dir}")
        logger.info(f"  Lookback Window: {self.lookback_seconds} seconds")
        logger.info(f"  Similarity Threshold: {self.similarity_threshold}")
    
    def find_video_file(self, video_id):
        """
        Find video file for given video_id
        
        Args:
            video_id: Video identifier
            
        Returns:
            Path to video file or None
        """
        # Try common extensions
        extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
        
        for ext in extensions:
            video_path = self.video_dir / f"{video_id}{ext}"
            if video_path.exists():
                return video_path
        
        return None
    
    def calculate_selection_score(self, frames_df):
        """
        Calculate selection score for each frame in the window
        
        Formula: Selection_Score = (1 - teacher_presence) * (1 + frame_position_weight)
        - Favors frames with less teacher
        - Favors frames closer to transition (higher index in window)
        
        Args:
            frames_df: DataFrame with frames in the search window
            
        Returns:
            DataFrame with selection_score column added
        """
        total_frames = len(frames_df)
        
        if total_frames == 0:
            return frames_df
        
        # Calculate frame position weight (0 to 1, with 1 being closest to transition)
        frames_df['frame_position_weight'] = np.arange(total_frames) / max(total_frames - 1, 1)
        
        # Calculate selection score
        frames_df['selection_score'] = (
            (1 - frames_df['teacher_presence']) * 
            (1 + frames_df['frame_position_weight'])
        )
        
        return frames_df
    
    def extract_frame_from_video(self, video_path, timestamp_seconds):
        """
        Extract a single frame from video at given timestamp
        
        Args:
            video_path: Path to video file
            timestamp_seconds: Timestamp in seconds
            
        Returns:
            numpy array (BGR image) or None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Set position to timestamp (in milliseconds)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            else:
                logger.warning(f"Failed to read frame at {timestamp_seconds:.2f}s")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return None
    
    def calculate_frame_similarity(self, img1, img2):
        """
        Calculate SSIM between two frames
        
        Args:
            img1: First image (BGR)
            img2: Second image (BGR)
            
        Returns:
            SSIM score (0 to 1)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize if dimensions don't match
        if gray1.shape != gray2.shape:
            h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))
        
        # Calculate SSIM
        similarity = ssim(gray1, gray2)
        return similarity
    
    def deduplicate_frames(self, extracted_frames):
        """
        Remove duplicate frames using SSIM
        If two consecutive frames are >92% similar, keep the one with lower teacher_presence
        
        Args:
            extracted_frames: List of dictionaries with frame info
            
        Returns:
            Deduplicated list of frames
        """
        if len(extracted_frames) <= 1:
            return extracted_frames
        
        logger.info("\n" + "="*60)
        logger.info("Deduplication Pass")
        logger.info("="*60)
        
        deduplicated = [extracted_frames[0]]
        duplicates_removed = 0
        
        for i in range(1, len(extracted_frames)):
            current = extracted_frames[i]
            previous = deduplicated[-1]
            
            # Load images
            current_img = cv2.imread(str(current['image_path']))
            previous_img = cv2.imread(str(previous['image_path']))
            
            # Calculate similarity
            similarity = self.calculate_frame_similarity(current_img, previous_img)
            
            if similarity >= self.similarity_threshold:
                # Duplicate detected - keep the one with lower teacher_presence
                if current['teacher_presence'] < previous['teacher_presence']:
                    # Current has less teacher - replace previous
                    logger.info(f"Duplicate: {previous['note_id']} vs {current['note_id']}")
                    logger.info(f"  SSIM: {similarity:.4f} (>= {self.similarity_threshold})")
                    logger.info(f"  Previous teacher: {previous['teacher_presence']:.3f}")
                    logger.info(f"  Current teacher:  {current['teacher_presence']:.3f}")
                    logger.info(f"  -> Keeping {current['note_id']} (lower teacher)")
                    
                    # Delete previous image
                    Path(previous['image_path']).unlink()
                    
                    # Replace in deduplicated list
                    deduplicated[-1] = current
                    duplicates_removed += 1
                else:
                    # Previous has less teacher - keep it, discard current
                    logger.info(f"Duplicate: {previous['note_id']} vs {current['note_id']}")
                    logger.info(f"  SSIM: {similarity:.4f} (>= {self.similarity_threshold})")
                    logger.info(f"  Previous teacher: {previous['teacher_presence']:.3f}")
                    logger.info(f"  Current teacher:  {current['teacher_presence']:.3f}")
                    logger.info(f"  -> Keeping {previous['note_id']} (lower teacher)")
                    
                    # Delete current image
                    Path(current['image_path']).unlink()
                    duplicates_removed += 1
            else:
                # Not a duplicate - keep both
                deduplicated.append(current)
        
        logger.info(f"\nDeduplication complete:")
        logger.info(f"  Original frames: {len(extracted_frames)}")
        logger.info(f"  Duplicates removed: {duplicates_removed}")
        logger.info(f"  Final frames: {len(deduplicated)}")
        
        return deduplicated
    
    def extract_notes_from_video(self, video_id):
        """
        Extract notes from a single video
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of extracted frame info dictionaries
        """
        logger.info("\n" + "="*60)
        logger.info(f"Processing: {video_id}")
        logger.info("="*60)
        
        # Find video file
        video_path = self.find_video_file(video_id)
        if video_path is None:
            logger.error(f"Video file not found for {video_id}")
            return []
        
        logger.info(f"Video: {video_path}")
        
        # Load feature CSV
        csv_path = self.csv_dir / f"{video_id}_features.csv"
        if not csv_path.exists():
            logger.error(f"CSV not found: {csv_path}")
            return []
        
        logger.info(f"CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} frames")
        
        # Find transitions (label == 1)
        transitions = df[df['label'] == 1].copy()
        logger.info(f"Found {len(transitions)} transition frames")
        
        if len(transitions) == 0:
            logger.warning("No transitions found in this video")
            return []
        
        # Group consecutive transitions into clusters
        transitions = transitions.sort_values('frame_index')
        transition_clusters = []
        current_cluster = [transitions.iloc[0]]
        
        for i in range(1, len(transitions)):
            current = transitions.iloc[i]
            prev = current_cluster[-1]
            
            # If frames are consecutive (within 5 frames), same cluster
            if current['frame_index'] - prev['frame_index'] <= 5:
                current_cluster.append(current)
            else:
                transition_clusters.append(current_cluster)
                current_cluster = [current]
        
        # Add last cluster
        if current_cluster:
            transition_clusters.append(current_cluster)
        
        logger.info(f"Grouped into {len(transition_clusters)} transition events")
        
        # Create output directory for this video
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(exist_ok=True)
        
        extracted_frames = []
        
        # Process each transition cluster
        for cluster_idx, cluster in enumerate(transition_clusters, 1):
            # Use the last frame in cluster as the transition point
            transition_frame = cluster[-1]
            transition_timestamp = transition_frame['timestamp_seconds']
            transition_frame_idx = int(transition_frame['frame_index'])
            
            logger.info(f"\n[{cluster_idx}/{len(transition_clusters)}] Transition at "
                       f"{transition_timestamp:.2f}s (frame {transition_frame_idx})")
            
            # Define search window (look back 10 seconds)
            window_start_time = max(0, transition_timestamp - self.lookback_seconds)
            
            # Get frames in the window
            window_frames = df[
                (df['timestamp_seconds'] >= window_start_time) &
                (df['timestamp_seconds'] <= transition_timestamp)
            ].copy()
            
            if len(window_frames) == 0:
                logger.warning("  No frames in search window")
                continue
            
            logger.info(f"  Search window: {window_start_time:.2f}s to {transition_timestamp:.2f}s")
            logger.info(f"  Frames in window: {len(window_frames)}")
            
            # Calculate selection scores
            window_frames = self.calculate_selection_score(window_frames)
            
            # Find frame with highest selection score
            best_frame = window_frames.loc[window_frames['selection_score'].idxmax()]
            best_timestamp = best_frame['timestamp_seconds']
            best_teacher = best_frame['teacher_presence']
            best_score = best_frame['selection_score']
            
            logger.info(f"  Best frame: {best_timestamp:.2f}s")
            logger.info(f"    Teacher presence: {best_teacher:.3f}")
            logger.info(f"    Selection score: {best_score:.3f}")
            
            # Extract frame from video
            frame_img = self.extract_frame_from_video(video_path, best_timestamp)
            
            if frame_img is None:
                logger.warning("  Failed to extract frame")
                continue
            
            # Save image
            note_id = f"{video_id}_note_{cluster_idx:03d}"
            image_path = video_output_dir / f"{note_id}.jpg"
            cv2.imwrite(str(image_path), frame_img)
            
            logger.info(f"  Saved: {image_path.name}")
            
            # Store frame info
            extracted_frames.append({
                'video_id': video_id,
                'note_id': note_id,
                'cluster_index': cluster_idx,
                'timestamp_seconds': best_timestamp,
                'teacher_presence': best_teacher,
                'selection_score': best_score,
                'image_path': str(image_path)
            })
        
        # Deduplicate frames
        if len(extracted_frames) > 1:
            extracted_frames = self.deduplicate_frames(extracted_frames)
        
        logger.info(f"\n{video_id} Complete:")
        logger.info(f"  Extracted {len(extracted_frames)} unique notes")
        
        return extracted_frames
    
    def extract_all_videos(self):
        """Extract notes from all videos with CSVs"""
        logger.info("\n" + "="*60)
        logger.info("Lecture Notes Extraction Pipeline")
        logger.info("="*60)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Find all feature CSVs
        csv_files = sorted(self.csv_dir.glob('*_features.csv'))
        logger.info(f"\nFound {len(csv_files)} CSV files")
        
        all_extracted_frames = []
        total_notes = 0
        
        for csv_file in csv_files:
            video_id = csv_file.stem.replace('_features', '')
            
            try:
                extracted = self.extract_notes_from_video(video_id)
                all_extracted_frames.extend(extracted)
                total_notes += len(extracted)
                
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}", exc_info=True)
                continue
        
        # Save summary CSV
        if all_extracted_frames:
            summary_df = pd.DataFrame(all_extracted_frames)
            summary_path = self.output_dir / 'extraction_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"\nSummary saved: {summary_path}")
        
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total videos processed: {len(csv_files)}")
        logger.info(f"Total notes extracted: {total_notes}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("\nNext Steps:")
        logger.info("1. Review extracted frames in data/extracted_notes/")
        logger.info("2. Check extraction_summary.csv for metadata")
        logger.info("3. Use these frames for OCR/note generation")
        
        return all_extracted_frames


def main():
    """Main execution function"""
    extractor = NotesExtractor()
    extractor.extract_all_videos()


if __name__ == '__main__':
    main()
