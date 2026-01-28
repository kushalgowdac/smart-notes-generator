"""
Extract Best Quality Slides - Content-First Approach with Board Detection
Uses intelligent scoring to find frames with maximum board visibility and text clarity.

Algorithm:
1. Adaptive Window: T_prev+1s to T_curr (handles rapid teaching)
2. Board Mask: HSV-based detection of blackboard/chalk areas
3. Text Sharpness: Canny edge detection for text clarity
4. Frame Scoring: (Board_Visibility * 0.6) + (Edge_Score * 0.3) + (Proximity * 0.1)
5. Best Frame Selection: Highest score wins
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BestSlideExtractor:
    """Extract best quality slides using content-first board detection"""
    
    def __init__(self, 
                 lectures_dir: str = 'data/lectures',
                 videos_dir: str = 'data/videos',
                 lookback_seconds: float = 10.0,
                 fps: int = 30):
        
        self.lectures_dir = Path(lectures_dir)
        self.videos_dir = Path(videos_dir)
        self.lookback_seconds = lookback_seconds
        self.fps = fps
        
        logger.info("Initialized BestSlideExtractor (Content-First Approach)")
        logger.info(f"Lookback window: {lookback_seconds}s")
        logger.info(f"Processing at: {fps} FPS")
    
    def detect_board_visibility(self, frame: np.ndarray) -> float:
        """
        Detect board/chalk visibility using HSV color masking
        Returns: Board visibility score (0.0 = no board, 1.0 = full board visible)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        total_pixels = height * width
        
        # Define HSV ranges for blackboard/greenboard/whiteboard
        # Range 1: Dark green/black (typical blackboard/greenboard)
        lower_dark = np.array([0, 0, 0], dtype=np.uint8)
        upper_dark = np.array([180, 255, 80], dtype=np.uint8)
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Range 2: Green board (common in education)
        lower_green = np.array([35, 40, 40], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Range 3: White/Light board with chalk/marker
        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 30, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine all board masks
        board_mask = cv2.bitwise_or(mask_dark, mask_green)
        board_mask = cv2.bitwise_or(board_mask, mask_white)
        
        # Calculate board visibility score
        board_pixels = cv2.countNonZero(board_mask)
        board_visibility_score = board_pixels / total_pixels
        
        return board_visibility_score
    
    def calculate_sharpness_score(self, frame: np.ndarray) -> float:
        """
        Calculate sharpness using Laplacian variance to detect motion blur.
        Higher score = sharper image (less blur from teacher/camera movement).
        
        Returns: normalized sharpness score (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize: typical sharp lecture frames have variance 100-1000
        # Blurry frames: < 50, Sharp frames: > 200
        normalized = min(laplacian_var / 200.0, 1.0)
        return normalized
    
    def calculate_brightness_quality(self, frame: np.ndarray) -> float:
        """
        Calculate brightness quality - penalize over/underexposed frames.
        Returns: quality score (0-1), where 1 = optimal brightness
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Optimal brightness: 80-150 (mid-range)
        # Penalize very dark (< 50) or very bright (> 200)
        if 80 <= mean_brightness <= 150:
            return 1.0
        elif mean_brightness < 80:
            # Dark frames: linear penalty
            return max(mean_brightness / 80.0, 0.3)
        else:  # > 150
            # Bright frames: linear penalty
            return max((255 - mean_brightness) / 105.0, 0.3)
    
    def calculate_edge_score(self, frame: np.ndarray) -> Tuple[float, int, float]:
        """
        Calculate edge score using Canny + distribution quality.
        Higher score = more text visible and well-distributed.
        
        Returns: (normalized_edge_score, raw_edge_count, distribution_score)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        # Lower thresholds to catch more text edges
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        
        # Count edge pixels
        edge_count = cv2.countNonZero(edges)
        
        # Normalize by frame size
        height, width = frame.shape[:2]
        total_pixels = height * width
        edge_score_normalized = edge_count / total_pixels
        
        # Calculate edge distribution quality (prefer evenly distributed text)
        # Divide frame into 9 regions (3x3 grid)
        grid_h, grid_w = height // 3, width // 3
        region_densities = []
        for i in range(3):
            for j in range(3):
                region = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                region_pixels = grid_h * grid_w
                density = cv2.countNonZero(region) / region_pixels if region_pixels > 0 else 0
                region_densities.append(density)
        
        # Distribution quality: prefer uniform distribution
        # Good slides have text across regions, not concentrated in one spot
        if len(region_densities) > 0 and np.mean(region_densities) > 0:
            std_dev = np.std(region_densities)
            mean_density = np.mean(region_densities)
            # Coefficient of variation (normalized)
            cv = std_dev / (mean_density + 1e-6)
            distribution_score = max(1.0 - (cv / 2.0), 0.3)  # Penalize high variation
        else:
            distribution_score = 0.5
        
        return edge_score_normalized, edge_count, distribution_score
    
    def calculate_proximity_bias(self, current_frame_time: float, 
                                 window_start: float, 
                                 window_end: float) -> float:
        """
        Calculate proximity bias: 0.0 at start → 1.0 at end
        Slight preference for frames closer to transition (more complete content)
        """
        if window_end == window_start:
            return 1.0
        
        # Linear interpolation from 0.0 to 1.0
        progress = (current_frame_time - window_start) / (window_end - window_start)
        
        return progress
    
    def score_frame(self, frame: np.ndarray, 
                   frame_time: float,
                   window_start: float,
                   window_end: float) -> Tuple[float, Dict[str, Any]]:
        """
        Score a single frame using advanced content-first approach.
        Returns: (final_score, scoring_details)
        
        Enhanced Formula:
        Quality = Sharpness × Brightness × sqrt(Board × Edges × Distribution)
        Final = Quality × (1 + Proximity × 0.15)
        
        This requires: board visibility + text + sharpness + good brightness + distribution
        """
        # 1. Detect board visibility
        board_visibility_score = self.detect_board_visibility(frame)
        
        # 2. Calculate edge score with distribution
        edge_score_normalized, edge_count, distribution_score = self.calculate_edge_score(frame)
        
        # 3. Calculate sharpness (motion blur detection)
        sharpness_score = self.calculate_sharpness_score(frame)
        
        # 4. Calculate brightness quality
        brightness_quality = self.calculate_brightness_quality(frame)
        
        # 5. Calculate proximity bias (reduced to 15% from 30%)
        proximity_bias = self.calculate_proximity_bias(frame_time, window_start, window_end)
        
        # 6. Content quality: requires board + text + distribution
        # Using geometric mean (cube root) to require all three
        content_score = (board_visibility_score * edge_score_normalized * distribution_score) ** (1/3)
        
        # 7. Overall quality: content + sharpness + brightness
        quality_score = content_score * sharpness_score * brightness_quality
        
        # 8. Final score with reduced proximity bias (15% vs 30%)
        # Quality matters more than position in window
        proximity_multiplier = 1.0 + (proximity_bias * 0.15)
        
        final_score = quality_score * proximity_multiplier
        
        # Scoring details for debugging
        scoring_details = {
            'final_score': final_score,
            'quality_score': quality_score,
            'content_score': content_score,
            'board_visibility_score': board_visibility_score,
            'edge_score_normalized': edge_score_normalized,
            'edge_count': edge_count,
            'distribution_score': distribution_score,
            'sharpness_score': sharpness_score,
            'brightness_quality': brightness_quality,
            'proximity_bias': proximity_bias,
            'proximity_multiplier': proximity_multiplier
        }
        
        return final_score, scoring_details
    
    def find_best_frame(self, 
                       video_path: Path,
                       t_curr: float,
                       t_prev: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Find best frame in adaptive window between t_prev and t_curr.
        
        Adaptive Window with Transition Buffer:
        - Start: Max of (T_curr - 10s) OR (T_prev + 1s)
        - End: T_curr - 1.0s (buffer to avoid fade-out/transition effects)
        
        The 1-second buffer ensures we capture slides BEFORE transition effects 
        (fade-out, wipe, blur) begin, avoiding unclear/fading content.
        
        Returns: (best_frame, best_timestamp, scoring_details)
        """
        # Define adaptive window with 1-second buffer before transition
        window_start = max(t_curr - self.lookback_seconds, t_prev + 1.0)
        window_end = t_curr - 1.0  # 1-second buffer to avoid fade-out effects
        
        # Ensure valid window (at least 1 second)
        if window_start >= window_end:
            window_start = max(0, t_curr - 2.0)  # Minimum 2s window
            window_end = max(window_start + 1.0, t_curr - 0.5)  # At least 1s window, end 0.5s before transition
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range
        start_frame = int(window_start * video_fps)
        end_frame = int(window_end * video_fps)
        
        best_score = -1
        best_frame = None
        best_timestamp = t_curr
        best_details = {}
        
        # Sample frames (process every frame for accuracy)
        frames_to_check = []
        for frame_num in range(start_frame, end_frame + 1):
            frame_time = frame_num / video_fps
            frames_to_check.append((frame_num, frame_time))
        
        # Process frames
        for frame_num, frame_time in frames_to_check:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Score frame
            score, scoring_details = self.score_frame(
                frame, frame_time, window_start, window_end
            )
            
            # Update best if this is better
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_timestamp = frame_time
                best_details = scoring_details.copy()
                best_details['window_start'] = window_start
                best_details['window_end'] = window_end
                best_details['frames_evaluated'] = len(frames_to_check)
        
        cap.release()
        
        if best_frame is None:
            # Fallback: extract at t_curr
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_MSEC, t_curr * 1000)
            ret, best_frame = cap.read()
            cap.release()
            
            best_timestamp = t_curr
            best_details = {
                'final_score': 0.0,
                'quality_score': 0.0,
                'content_score': 0.0,
                'board_visibility_score': 0.0,
                'edge_score_normalized': 0.0,
                'edge_count': 0,
                'distribution_score': 0.0,
                'sharpness_score': 0.0,
                'brightness_quality': 0.0,
                'proximity_bias': 1.0,
                'proximity_multiplier': 1.0,
                'window_start': window_start,
                'window_end': window_end,
                'frames_evaluated': 0,
                'fallback': True
            }
        
        return best_frame, best_timestamp, best_details
    
    def extract_video_slides(self, video_id: str) -> Dict[str, Any]:
        """Extract all best slides for a single video"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Extracting slides: {video_id}")
        logger.info(f"{'='*70}")
        
        # Load transitions.json
        lecture_dir = self.lectures_dir / video_id
        transitions_file = lecture_dir / 'transitions.json'
        
        if not transitions_file.exists():
            logger.error(f"transitions.json not found: {transitions_file}")
            return None
        
        with open(transitions_file, 'r') as f:
            transitions_data = json.load(f)
        
        # Find video file
        video_file = None
        for ext in ['.mp4', '.mkv', '.avi', '.mov']:
            potential_file = self.videos_dir / f"{video_id}{ext}"
            if potential_file.exists():
                video_file = potential_file
                break
        
        if not video_file:
            logger.error(f"Video file not found for {video_id}")
            return None
        
        logger.info(f"Video: {video_file.name}")
        logger.info(f"Total transitions: {transitions_data['total_transitions']}")
        
        # Process ALL transitions (including blanks)
        # The adaptive window will look backward to find good content before blank transitions
        all_transitions = transitions_data['transitions']
        blank_count = sum(1 for t in all_transitions if t.get('is_blank', False))
        
        logger.info(f"Processing all transitions: {len(all_transitions)}")
        logger.info(f"  - Blank transition points: {blank_count} (will look backward for content)")
        logger.info(f"  - Content transition points: {len(all_transitions) - blank_count}")
        
        # Create output directories
        slides_dir = lecture_dir / 'slides'
        slides_dir.mkdir(exist_ok=True)
        
        # Extract slides with progress bar
        metadata_entries = []
        
        with tqdm(total=len(all_transitions), desc=f"Extracting {video_id}", unit="slide") as pbar:
            for i, transition in enumerate(all_transitions):
                t_curr = transition['timestamp']
                
                # Determine t_prev
                if i == 0:
                    t_prev = 0.0
                else:
                    # Get previous transition timestamp (regardless of blank status)
                    t_prev = all_transitions[i-1]['timestamp']
                
                # Find best frame
                try:
                    best_frame, best_timestamp, scoring_details = self.find_best_frame(
                        video_file, t_curr, t_prev
                    )
                    
                    # Save slide (high quality PNG)
                    slide_filename = f"{video_id}_slide_{transition['index']:03d}.png"
                    slide_path = slides_dir / slide_filename
                    
                    cv2.imwrite(str(slide_path), best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    
                    # Create metadata entry
                    metadata_entry = {
                        'slide_number': transition['index'],
                        'filename': slide_filename,
                        'capture_timestamp': round(best_timestamp, 2),
                        'capture_timestamp_formatted': self._format_timestamp(best_timestamp),
                        'transition_timestamp': t_curr,
                        'transition_was_blank': transition.get('is_blank', False),  # Track if transition point was blank
                        'audio_start_time': round(t_prev, 2),
                        'audio_end_time': round(t_curr, 2),
                        'audio_duration': round(t_curr - t_prev, 2),
                        'audio_filename': f"{video_id}_slide_{transition['index']:03d}.mp3",
                        'ocr_status': 'pending',
                        'ai_analysis_status': 'pending',
                        'scoring': {
                            'final_score': round(scoring_details['final_score'], 4),
                            'quality_score': round(scoring_details['quality_score'], 4),
                            'content_score': round(scoring_details['content_score'], 4),
                            'board_visibility_score': round(scoring_details['board_visibility_score'], 4),
                            'edge_score_normalized': round(scoring_details['edge_score_normalized'], 4),
                            'edge_count': scoring_details['edge_count'],
                            'distribution_score': round(scoring_details['distribution_score'], 4),
                            'sharpness_score': round(scoring_details['sharpness_score'], 4),
                            'brightness_quality': round(scoring_details['brightness_quality'], 4),
                            'proximity_bias': round(scoring_details['proximity_bias'], 4),
                            'proximity_multiplier': round(scoring_details['proximity_multiplier'], 4),
                            'window_start': round(scoring_details['window_start'], 2),
                            'window_end': round(scoring_details['window_end'], 2),
                            'frames_evaluated': scoring_details['frames_evaluated']
                        },
                        'original_transition_data': {
                            'edge_count': transition.get('edge_count'),
                            'has_audio': transition.get('has_audio'),
                            'is_blank_at_transition': transition.get('is_blank', False)
                        }
                    }
                    
                    metadata_entries.append(metadata_entry)
                    
                    pbar.set_postfix({
                        'score': f"{scoring_details['final_score']:.3f}",
                        'quality': f"{scoring_details['quality_score']:.3f}",
                        'sharp': f"{scoring_details['sharpness_score']:.2f}",
                        'edges': scoring_details['edge_count']
                    })
                    
                except Exception as e:
                    logger.error(f"Error extracting slide {transition['index']}: {e}")
                
                pbar.update(1)
        
        # Create metadata.json
        metadata = {
            'video_id': video_id,
            'generated_at': datetime.now().isoformat(),
            'extraction_method': 'content_first_board_detection',
            'total_slides': len(metadata_entries),
            'extraction_params': {
                'lookback_seconds': self.lookback_seconds,
                'fps': self.fps,
                'board_detection': 'hsv_color_masking',
                'edge_detection': 'canny',
                'scoring_formula': '(Board*0.6) + (Edges*0.3) + (Proximity*0.1)'
            },
            'slides': metadata_entries
        }
        
        metadata_file = lecture_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n✅ Extracted {len(metadata_entries)} slides")
        logger.info(f"   Slides saved to: {slides_dir}")
        logger.info(f"   Metadata saved to: {metadata_file}")
        
        return metadata
    
    def extract_all_videos(self, video_ids: List[str] = None) -> Dict[str, Any]:
        """Extract slides for all videos or specified list"""
        
        logger.info("\n" + "="*70)
        logger.info("EXTRACTING BEST SLIDES - ALL VIDEOS")
        logger.info("="*70)
        
        # Get all video folders if not specified
        if video_ids is None:
            video_ids = [d.name for d in self.lectures_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Processing {len(video_ids)} videos")
        
        results = {
            'total_videos': len(video_ids),
            'successful': 0,
            'failed': 0,
            'total_slides': 0,
            'videos': []
        }
        
        for video_id in video_ids:
            try:
                metadata = self.extract_video_slides(video_id)
                
                if metadata:
                    results['successful'] += 1
                    results['total_slides'] += metadata['total_slides']
                    results['videos'].append({
                        'video_id': video_id,
                        'status': 'success',
                        'slides': metadata['total_slides']
                    })
                else:
                    results['failed'] += 1
                    results['videos'].append({
                        'video_id': video_id,
                        'status': 'failed',
                        'slides': 0
                    })
            
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                results['failed'] += 1
                results['videos'].append({
                    'video_id': video_id,
                    'status': 'error',
                    'slides': 0,
                    'error': str(e)
                })
        
        # Save summary
        summary_file = self.lectures_dir / 'extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info("EXTRACTION COMPLETE!")
        logger.info("="*70)
        logger.info(f"Successful: {results['successful']}/{results['total_videos']}")
        logger.info(f"Failed: {results['failed']}/{results['total_videos']}")
        logger.info(f"Total slides extracted: {results['total_slides']}")
        logger.info(f"Summary saved: {summary_file}")
        
        return results
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract best quality slides with adaptive window')
    parser.add_argument('--lectures-dir', default='data/lectures', help='Lectures directory')
    parser.add_argument('--videos-dir', default='data/videos', help='Videos directory')
    parser.add_argument('--lookback', type=float, default=10.0, help='Max lookback window (seconds)')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--video-id', help='Process single video only')
    
    args = parser.parse_args()
    
    extractor = BestSlideExtractor(
        lectures_dir=args.lectures_dir,
        videos_dir=args.videos_dir,
        lookback_seconds=args.lookback,
        fps=args.fps
    )
    
    if args.video_id:
        # Process single video
        extractor.extract_video_slides(args.video_id)
    else:
        # Process all videos
        extractor.extract_all_videos()


if __name__ == '__main__':
    main()
