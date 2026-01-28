"""
File: video_feature_extractor.py
Purpose: Extract features from lecture videos for transition detection using OpenCV, Scikit-Image, and Pandas
Author: Smart Notes Generator Team
Created: 2026-01-25
Last Modified: 2026-01-26

This script processes lecture videos at 5 FPS and extracts specialized features including:
- Teacher masking (black pixel ratio + skin detection)
- Teacher presence (combined black + skin pixels)
- Global differences (SSIM, MSE, histogram correlation)
- Edge analysis with Canny detection
- Tri-zonal ROI analysis
- Temporal memory with sliding windows
"""

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """
    Extract comprehensive features from lecture videos for transition detection.
    
    Features include:
    - Black pixel ratio (teacher's black t-shirt)
    - Skin pixel ratio (teacher's face/hands using HSV detection)
    - Teacher presence (combined black + skin pixels)
    - Global SSIM, MSE, and histogram correlation
    - Edge detection metrics
    - Tri-zonal ROI analysis
    - Temporal sliding window features
    """
    
    def __init__(self, video_path: str, output_dir: str = "data/output", target_fps: int = 5):
        """
        Initialize the feature extractor.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save output CSV files
            target_fps: Target frame rate for processing (default: 5 FPS)
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.window_size = 10  # 2 seconds at 5 FPS
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate video file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        logger.info(f"Initialized VideoFeatureExtractor for: {self.video_path.name}")
    
    def calculate_black_pixel_ratio(self, frame: np.ndarray) -> float:
        """
        Calculate the ratio of near-black pixels (teacher's black t-shirt).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Ratio of black pixels (0.0 to 1.0)
        """
        # Identify pixels where all RGB values < 50
        black_mask = np.all(frame < 50, axis=2)
        black_pixel_count = np.sum(black_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        
        return black_pixel_count / total_pixels if total_pixels > 0 else 0.0
    
    def calculate_skin_pixel_ratio(self, frame: np.ndarray) -> float:
        """
        Calculate the ratio of skin-colored pixels (teacher's face/hands).
        Uses HSV color space for robust skin detection.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Ratio of skin pixels (0.0 to 1.0)
        """
        try:
            # Convert BGR to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            # Lower: [H=0, S=20, V=70]
            # Upper: [H=20, S=255, V=255]
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
            
            # Calculate skin pixel ratio
            skin_pixel_count = np.sum(skin_mask > 0)
            total_pixels = frame.shape[0] * frame.shape[1]
            
            return skin_pixel_count / total_pixels if total_pixels > 0 else 0.0
        except Exception as e:
            logger.warning(f"Skin detection failed: {e}")
            return 0.0
    
    def calculate_global_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two frames.
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            
        Returns:
            SSIM value (-1.0 to 1.0)
        """
        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if len(frame2.shape) == 3:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            return ssim(frame1, frame2, data_range=255)
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.0
    
    def calculate_mse(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate Mean Squared Error between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            MSE value
        """
        try:
            return mse(frame1, frame2)
        except Exception as e:
            logger.warning(f"MSE calculation failed: {e}")
            return 0.0
    
    def calculate_histogram_correlation(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate histogram correlation between two frames.
        
        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            
        Returns:
            Correlation coefficient (0.0 to 1.0)
        """
        try:
            # Calculate histograms for each channel
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Calculate correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return max(0.0, correlation)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Histogram correlation calculation failed: {e}")
            return 0.0
    
    def calculate_edge_features(self, frame: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Calculate edge features using Canny edge detection.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (edge_count, edge_image)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels
            edge_count = np.sum(edges > 0)
            
            return edge_count, edges
        except Exception as e:
            logger.warning(f"Edge detection failed: {e}")
            return 0, np.zeros_like(frame[:, :, 0])
    
    def calculate_tri_zonal_roi(self, frame1: np.ndarray, frame2: np.ndarray, 
                                edges1: np.ndarray, edges2: np.ndarray) -> Dict[str, float]:
        """
        Calculate features for three zones in the bottom 20% of the frame.
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            edges1: Edge image of first frame
            edges2: Edge image of second frame
            
        Returns:
            Dictionary with zonal features
        """
        features = {}
        
        try:
            height, width = frame1.shape[:2]
            
            # Define bottom 20% region
            bottom_start = int(height * 0.8)
            bottom_region_height = height - bottom_start
            
            # Divide into 3 equal zones
            zone_width = width // 3
            zones = ['left', 'center', 'right']
            
            for i, zone_name in enumerate(zones):
                # Extract zone boundaries
                x_start = i * zone_width
                x_end = (i + 1) * zone_width if i < 2 else width
                
                # Extract zone regions
                zone1 = frame1[bottom_start:, x_start:x_end]
                zone2 = frame2[bottom_start:, x_start:x_end]
                edges_zone1 = edges1[bottom_start:, x_start:x_end]
                edges_zone2 = edges2[bottom_start:, x_start:x_end]
                
                # Calculate local SSIM
                zone_ssim = self.calculate_global_ssim(zone1, zone2)
                features[f'zone_{zone_name}_ssim'] = zone_ssim
                
                # Calculate edge density
                total_pixels = zone1.shape[0] * zone1.shape[1]
                edge_density1 = np.sum(edges_zone1 > 0) / total_pixels if total_pixels > 0 else 0
                edge_density2 = np.sum(edges_zone2 > 0) / total_pixels if total_pixels > 0 else 0
                
                features[f'zone_{zone_name}_edge_density'] = (edge_density1 + edge_density2) / 2
                
        except Exception as e:
            logger.warning(f"Tri-zonal ROI calculation failed: {e}")
            for zone in ['left', 'center', 'right']:
                features[f'zone_{zone}_ssim'] = 0.0
                features[f'zone_{zone}_edge_density'] = 0.0
        
        return features
    
    def calculate_edge_decay_velocity(self, edge_counts: list) -> float:
        """
        Calculate the slope of edge count over the sliding window.
        
        Args:
            edge_counts: List of edge counts in the window
            
        Returns:
            Slope (edge decay velocity)
        """
        if len(edge_counts) < 2:
            return 0.0
        
        try:
            # Simple linear regression to calculate slope
            n = len(edge_counts)
            x = np.arange(n)
            y = np.array(edge_counts)
            
            # Calculate slope using least squares
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            
            return slope
        except Exception as e:
            logger.warning(f"Edge decay velocity calculation failed: {e}")
            return 0.0
    
    def extract_features(self) -> pd.DataFrame:
        """
        Extract all features from the video.
        
        Returns:
            DataFrame with all extracted features
        """
        logger.info(f"Starting feature extraction for: {self.video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties - FPS: {original_fps}, Total frames: {total_frames}")
        
        # Calculate frame skip to achieve target FPS
        frame_skip = max(1, int(original_fps / self.target_fps))
        
        # Storage for features
        features_list = []
        
        # Temporal window storage
        ssim_window = []
        edge_count_window = []
        
        # Previous frame storage
        prev_frame = None
        prev_edges = None
        
        frame_index = 0
        processed_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames to achieve target FPS
                if frame_index % frame_skip != 0:
                    frame_index += 1
                    continue
                
                # Calculate timestamp
                timestamp_seconds = frame_index / original_fps
                
                # Convert to grayscale for some calculations
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Initialize feature dictionary
                feature_dict = {
                    'video_id': self.video_path.stem,
                    'frame_index': processed_count,
                    'timestamp_seconds': timestamp_seconds
                }
                
                # Calculate edge features for current frame
                edge_count, edges = self.calculate_edge_features(frame)
                
                if prev_frame is not None:
                    # Calculate black pixel ratio
                    feature_dict['black_pixel_ratio'] = self.calculate_black_pixel_ratio(frame)
                    
                    # Calculate skin pixel ratio
                    feature_dict['skin_pixel_ratio'] = self.calculate_skin_pixel_ratio(frame)
                    
                    # Calculate teacher presence (combined feature)
                    feature_dict['teacher_presence'] = (feature_dict['black_pixel_ratio'] + 
                                                       feature_dict['skin_pixel_ratio'])
                    
                    # Calculate global differences
                    feature_dict['global_ssim'] = self.calculate_global_ssim(gray_frame, prev_frame)
                    feature_dict['global_mse'] = self.calculate_mse(gray_frame, prev_frame)
                    feature_dict['histogram_correlation'] = self.calculate_histogram_correlation(frame, 
                                                                                                  cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR) if len(prev_frame.shape) == 2 else prev_frame)
                    
                    # Calculate edge change rate
                    prev_edge_count = np.sum(prev_edges > 0) if prev_edges is not None else 0
                    feature_dict['edge_count'] = edge_count
                    feature_dict['edge_change_rate'] = edge_count - prev_edge_count
                    
                    # Calculate tri-zonal ROI features
                    zonal_features = self.calculate_tri_zonal_roi(prev_frame, gray_frame, 
                                                                   prev_edges, edges)
                    feature_dict.update(zonal_features)
                    
                    # Update temporal windows
                    ssim_window.append(feature_dict['global_ssim'])
                    edge_count_window.append(edge_count)
                    
                    # Keep window size limited
                    if len(ssim_window) > self.window_size:
                        ssim_window.pop(0)
                        edge_count_window.pop(0)
                    
                    # Calculate rolling statistics
                    if len(ssim_window) > 0:
                        feature_dict['ssim_rolling_mean'] = np.mean(ssim_window)
                        feature_dict['ssim_rolling_std'] = np.std(ssim_window)
                        feature_dict['ssim_rolling_max'] = np.max(ssim_window)
                        
                        feature_dict['edge_rolling_mean'] = np.mean(edge_count_window)
                        feature_dict['edge_rolling_std'] = np.std(edge_count_window)
                        feature_dict['edge_rolling_max'] = np.max(edge_count_window)
                        
                        # Calculate edge decay velocity
                        feature_dict['edge_decay_velocity'] = self.calculate_edge_decay_velocity(edge_count_window)
                    else:
                        # Initialize with zeros for first frame
                        feature_dict['ssim_rolling_mean'] = 0.0
                        feature_dict['ssim_rolling_std'] = 0.0
                        feature_dict['ssim_rolling_max'] = 0.0
                        feature_dict['edge_rolling_mean'] = 0.0
                        feature_dict['edge_rolling_std'] = 0.0
                        feature_dict['edge_rolling_max'] = 0.0
                        feature_dict['edge_decay_velocity'] = 0.0
                    
                else:
                    # First frame - initialize all features to 0
                    feature_dict['black_pixel_ratio'] = 0.0
                    feature_dict['skin_pixel_ratio'] = 0.0
                    feature_dict['teacher_presence'] = 0.0
                    feature_dict['global_ssim'] = 0.0
                    feature_dict['global_mse'] = 0.0
                    feature_dict['histogram_correlation'] = 0.0
                    feature_dict['edge_count'] = edge_count
                    feature_dict['edge_change_rate'] = 0.0
                    
                    # Zonal features
                    for zone in ['left', 'center', 'right']:
                        feature_dict[f'zone_{zone}_ssim'] = 0.0
                        feature_dict[f'zone_{zone}_edge_density'] = 0.0
                    
                    # Rolling statistics
                    feature_dict['ssim_rolling_mean'] = 0.0
                    feature_dict['ssim_rolling_std'] = 0.0
                    feature_dict['ssim_rolling_max'] = 0.0
                    feature_dict['edge_rolling_mean'] = 0.0
                    feature_dict['edge_rolling_std'] = 0.0
                    feature_dict['edge_rolling_max'] = 0.0
                    feature_dict['edge_decay_velocity'] = 0.0
                
                # Add label column (initialized to 0)
                feature_dict['label'] = 0
                
                features_list.append(feature_dict)
                
                # Update previous frame
                prev_frame = gray_frame.copy()
                prev_edges = edges.copy()
                
                processed_count += 1
                
                # Log progress
                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count} frames...")
                
                frame_index += 1
                
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            raise
        finally:
            cap.release()
        
        logger.info(f"Feature extraction complete. Processed {processed_count} frames.")
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Handle first 10 frames by padding with the first valid calculation
        if len(df) > 10:
            first_valid_idx = 10
            for col in df.columns:
                if col not in ['video_id', 'frame_index', 'timestamp_seconds', 'label']:
                    # Pad first 10 rows with value from frame 10
                    df.loc[:9, col] = df.loc[first_valid_idx, col]
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Min-Max scaling to all features (per video).
        
        Args:
            df: DataFrame with extracted features
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Applying Min-Max normalization...")
        
        # Columns to exclude from normalization
        exclude_cols = ['video_id', 'frame_index', 'timestamp_seconds', 'label']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Apply Min-Max scaling
        for col in feature_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Avoid division by zero
            if max_val - min_val > 1e-10:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
        
        logger.info("Normalization complete.")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame) -> Path:
        """
        Save the feature DataFrame to CSV.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Path to saved CSV file
        """
        # Generate output filename
        output_filename = f"{self.video_path.stem}_features.csv"
        output_path = self.output_dir / output_filename
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Features saved to: {output_path}")
        
        return output_path
    
    def process(self) -> Path:
        """
        Main processing pipeline: extract, normalize, and save features.
        
        Returns:
            Path to output CSV file
        """
        try:
            # Extract features
            df = self.extract_features()
            
            # Normalize features
            df = self.normalize_features(df)
            
            # Save to CSV
            output_path = self.save_to_csv(df)
            
            logger.info(f"Processing complete for {self.video_path.name}")
            logger.info(f"Total features extracted: {len(df)} frames, {len(df.columns)} columns")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Processing failed for {self.video_path.name}: {e}")
            raise


def process_video_folder(input_folder: str = "data/videos", output_folder: str = "data/output", 
                         target_fps: int = 5):
    """
    Process all videos in a folder.
    
    Args:
        input_folder: Folder containing input videos
        output_folder: Folder to save output CSV files
        target_fps: Target frame rate for processing
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        logger.error(f"Input folder not found: {input_folder}")
        return
    
    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'*{ext}'))
        video_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not video_files:
        logger.warning(f"No video files found in {input_folder}")
        return
    
    logger.info(f"Found {len(video_files)} video(s) to process")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing video {i}/{len(video_files)}: {video_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            extractor = VideoFeatureExtractor(
                video_path=str(video_path),
                output_dir=output_folder,
                target_fps=target_fps
            )
            
            output_path = extractor.process()
            logger.info(f"[SUCCESS] Successfully processed: {video_path.name}")
            
        except Exception as e:
            logger.error(f"[FAILED] Failed to process {video_path.name}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch processing complete!")
    logger.info(f"{'='*60}")


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from lecture videos for transition detection')
    parser.add_argument('--input', '-i', type=str, default='data/videos',
                       help='Input folder containing videos (default: data/videos)')
    parser.add_argument('--output', '-o', type=str, default='data/output',
                       help='Output folder for CSV files (default: data/output)')
    parser.add_argument('--fps', '-f', type=int, default=5,
                       help='Target FPS for processing (default: 5)')
    parser.add_argument('--single', '-s', type=str, default=None,
                       help='Process a single video file instead of a folder')
    
    args = parser.parse_args()
    
    # Process single video or entire folder
    if args.single:
        logger.info(f"Processing single video: {args.single}")
        extractor = VideoFeatureExtractor(
            video_path=args.single,
            output_dir=args.output,
            target_fps=args.fps
        )
        extractor.process()
    else:
        logger.info(f"Processing all videos in folder: {args.input}")
        process_video_folder(
            input_folder=args.input,
            output_folder=args.output,
            target_fps=args.fps
        )


if __name__ == "__main__":
    main()
