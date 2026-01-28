"""
Smart Notes Generator - Master Pipeline
Process a new lecture video end-to-end with 98.68% F1-score pipeline
"""

import cv2
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import logging
import argparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartNotesProcessor:
    """End-to-end pipeline for processing new lecture videos"""
    
    def __init__(self, video_path: str, output_base_dir='data/lectures',
                 model_path='models/xgboost_model_20260126_160645.pkl',
                 prediction_threshold=0.01,  # 1% threshold - model trained on rare transitions (11 frames)
                 ssim_dedup_threshold=0.95,  # 95% threshold (board-only comparison, ignores teacher movement)
                 blank_edge_threshold=0.02,  # 2% threshold - HD videos have low edge density
                 lookback_seconds=10.0,
                 feature_extraction_fps=5,  # 5fps for feature extraction (matches training)
                 slide_extraction_fps=30):  # 30fps for best slide extraction
        
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.video_id = self.video_path.stem
        self.output_base = Path(output_base_dir)
        self.lecture_dir = self.output_base / self.video_id
        
        # Load trained model
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Try loading with pickle first, then joblib
        try:
            import pickle
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except:
            self.model = joblib.load(self.model_path)
        
        # Parameters (same as training/testing)
        self.prediction_threshold = prediction_threshold
        self.ssim_dedup_threshold = ssim_dedup_threshold
        self.blank_edge_threshold = blank_edge_threshold
        self.lookback_seconds = lookback_seconds
        self.feature_extraction_fps = feature_extraction_fps  # 5fps for transition detection
        self.slide_extraction_fps = slide_extraction_fps  # 30fps for best slides
        
        # Create directory structure
        self._create_directories()
        
        logger.info("="*70)
        logger.info("SMART NOTES GENERATOR - MASTER PIPELINE")
        logger.info("="*70)
        logger.info(f"Video: {self.video_path.name}")
        logger.info(f"Output: {self.lecture_dir}")
        logger.info(f"Model: {self.model_path.name}")
        logger.info(f"Feature Extraction FPS: {self.feature_extraction_fps} (transition detection)")
        logger.info(f"Slide Extraction FPS: {self.slide_extraction_fps} (best quality frames)")
        logger.info(f"Prediction Threshold: {self.prediction_threshold}")
        logger.info(f"SSIM Dedup Threshold: {self.ssim_dedup_threshold}")
    
    def _create_directories(self):
        """Create output directory structure (removes old data if exists)"""
        # If directory exists, backup and remove old data
        if self.lecture_dir.exists():
            logger.warning(f"⚠️ Output directory already exists: {self.lecture_dir}")
            
            # Create backup directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.lecture_dir.parent / f"{self.video_id}_backup_{timestamp}"
            
            logger.info(f"📦 Creating backup: {backup_dir.name}")
            shutil.copytree(self.lecture_dir, backup_dir)
            logger.info(f"✅ Backup created successfully")
            
            # Remove old directory
            logger.info(f"🗑️ Removing old data: {self.lecture_dir.name}")
            shutil.rmtree(self.lecture_dir)
            logger.info(f"✅ Old data removed")
        
        # Create fresh directory structure
        logger.info(f"📁 Creating fresh directory structure")
        self.lecture_dir.mkdir(parents=True, exist_ok=True)
        (self.lecture_dir / 'audio').mkdir(exist_ok=True)
        (self.lecture_dir / 'slides').mkdir(exist_ok=True)
        (self.lecture_dir / 'transition_previews').mkdir(exist_ok=True)
        logger.info(f"✅ Directory structure ready")
    
    def extract_audio(self):
        """Extract audio from video using OpenCV"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: AUDIO EXTRACTION")
        logger.info("="*70)
        
        # Note: OpenCV doesn't extract audio directly
        # Using ffmpeg would be better, but for now we'll skip or use subprocess
        audio_path = self.lecture_dir / 'audio' / f"{self.video_id}.wav"
        
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(self.video_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                str(audio_path), '-y'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Audio extracted: {audio_path.name}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"⚠️ Audio extraction failed (ffmpeg required): {e}")
            logger.warning("   Continuing without audio...")
    
    def extract_features_and_predict(self):
        """
        STEP 2: Feature Extraction & Prediction
        Extract features from video and run model inference
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: FEATURE EXTRACTION & PREDICTION")
        logger.info("="*70)
        
        cap = cv2.VideoCapture(str(self.video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        logger.info(f"Video FPS: {video_fps:.2f}")
        logger.info(f"Total Frames: {total_frames}")
        logger.info(f"Duration: {duration:.2f}s")
        
        # Storage for features
        features_list = []
        previous_frame = None
        previous_gray = None
        previous_edges = None
        
        # Rolling windows for features (window size = 5)
        window_size = 5
        ssim_window = []
        edge_window = []
        
        # Calculate frame sampling interval for 5fps
        frame_interval = int(video_fps / self.feature_extraction_fps)
        sampled_frames = total_frames // frame_interval
        
        logger.info(f"Sampling Strategy: {self.feature_extraction_fps}fps (every {frame_interval} frames)")
        logger.info(f"Expected Sampled Frames: ~{sampled_frames}")
        
        pbar = tqdm(total=sampled_frames, desc="Extracting features", unit="frame")
        
        frame_idx = 0
        sampled_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames at the target fps interval
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            timestamp = frame_idx / video_fps
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Feature 1: Black pixel ratio (for dark backgrounds)
            black_pixels = np.sum(gray < 50)
            total_pixels = gray.shape[0] * gray.shape[1]
            black_pixel_ratio = black_pixels / total_pixels
            
            # Feature 2-3: Skin detection (teacher presence)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Light skin
            lower_skin_light = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin_light = np.array([20, 150, 255], dtype=np.uint8)
            mask_light = cv2.inRange(hsv, lower_skin_light, upper_skin_light)
            
            # Dark skin
            lower_skin_dark = np.array([0, 20, 0], dtype=np.uint8)
            upper_skin_dark = np.array([20, 150, 120], dtype=np.uint8)
            mask_dark = cv2.inRange(hsv, lower_skin_dark, upper_skin_dark)
            
            skin_mask = cv2.bitwise_or(mask_light, mask_dark)
            skin_pixels = cv2.countNonZero(skin_mask)
            skin_pixel_ratio = skin_pixels / total_pixels
            teacher_presence = skin_pixel_ratio
            
            # Feature 4-6: SSIM with previous frame
            if previous_gray is not None:
                if gray.shape == previous_gray.shape:
                    global_ssim = ssim(gray, previous_gray)
                else:
                    global_ssim = 1.0
                
                # MSE
                global_mse = np.mean((gray.astype(float) - previous_gray.astype(float)) ** 2)
            else:
                global_ssim = 1.0
                global_mse = 0.0
            
            # Feature 7: Histogram correlation
            hist_current = cv2.calcHist([gray], [0], None, [256], [0, 256])
            if previous_gray is not None:
                hist_prev = cv2.calcHist([previous_gray], [0], None, [256], [0, 256])
                histogram_correlation = cv2.compareHist(hist_current, hist_prev, cv2.HISTCMP_CORREL)
            else:
                histogram_correlation = 1.0
            
            # Feature 8-9: Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_count = cv2.countNonZero(edges) / total_pixels
            
            if previous_edges is not None:
                edge_change_rate = abs(edge_count - (cv2.countNonZero(previous_edges) / total_pixels))
            else:
                edge_change_rate = 0.0
            
            # Feature 10-16: Zone-based SSIM and edge density (3 zones: left, center, right)
            h, w = gray.shape
            zone_width = w // 3
            
            zones = {
                'left': (0, zone_width),
                'center': (zone_width, 2*zone_width),
                'right': (2*zone_width, w)
            }
            
            zone_ssims = {}
            zone_edge_densities = {}
            
            for zone_name, (start, end) in zones.items():
                zone_gray = gray[:, start:end]
                zone_edges = edges[:, start:end]
                
                if previous_gray is not None:
                    zone_prev = previous_gray[:, start:end]
                    if zone_gray.shape == zone_prev.shape:
                        zone_ssim = ssim(zone_gray, zone_prev)
                    else:
                        zone_ssim = 1.0
                else:
                    zone_ssim = 1.0
                
                zone_edge_density = cv2.countNonZero(zone_edges) / (zone_gray.shape[0] * zone_gray.shape[1])
                
                zone_ssims[zone_name] = zone_ssim
                zone_edge_densities[zone_name] = zone_edge_density
            
            # Update rolling windows
            ssim_window.append(global_ssim)
            edge_window.append(edge_count)
            
            if len(ssim_window) > window_size:
                ssim_window.pop(0)
            if len(edge_window) > window_size:
                edge_window.pop(0)
            
            # Feature 17-22: Rolling statistics
            if len(ssim_window) > 0:
                ssim_rolling_mean = np.mean(ssim_window)
                ssim_rolling_std = np.std(ssim_window) if len(ssim_window) > 1 else 0.0
                ssim_rolling_max = np.max(ssim_window)
            else:
                ssim_rolling_mean = 1.0
                ssim_rolling_std = 0.0
                ssim_rolling_max = 1.0
            
            if len(edge_window) > 0:
                edge_rolling_mean = np.mean(edge_window)
                edge_rolling_std = np.std(edge_window) if len(edge_window) > 1 else 0.0
                edge_rolling_max = np.max(edge_window)
            else:
                edge_rolling_mean = 0.0
                edge_rolling_std = 0.0
                edge_rolling_max = 0.0
            
            # Feature 23: Edge decay velocity
            if len(edge_window) >= 2:
                edge_decay_velocity = edge_window[-1] - edge_window[-2]
            else:
                edge_decay_velocity = 0.0
            
            # Store all 21 features
            features_list.append({
                'frame_number': frame_idx,  # Actual frame number at 30fps
                'sampled_index': sampled_idx,  # Index in sampled sequence
                'timestamp_seconds': round(timestamp, 2),
                'black_pixel_ratio': black_pixel_ratio,
                'skin_pixel_ratio': skin_pixel_ratio,
                'teacher_presence': teacher_presence,
                'global_ssim': global_ssim,
                'global_mse': global_mse,
                'histogram_correlation': histogram_correlation,
                'edge_count': edge_count,
                'edge_change_rate': edge_change_rate,
                'zone_left_ssim': zone_ssims['left'],
                'zone_left_edge_density': zone_edge_densities['left'],
                'zone_center_ssim': zone_ssims['center'],
                'zone_center_edge_density': zone_edge_densities['center'],
                'zone_right_ssim': zone_ssims['right'],
                'zone_right_edge_density': zone_edge_densities['right'],
                'ssim_rolling_mean': ssim_rolling_mean,
                'ssim_rolling_std': ssim_rolling_std,
                'ssim_rolling_max': ssim_rolling_max,
                'edge_rolling_mean': edge_rolling_mean,
                'edge_rolling_std': edge_rolling_std,
                'edge_rolling_max': edge_rolling_max,
                'edge_decay_velocity': edge_decay_velocity
            })
            
            previous_frame = frame.copy()
            previous_gray = gray.copy()
            previous_edges = edges.copy()
            frame_idx += 1
            sampled_idx += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Save features
        features_path = self.lecture_dir / f"{self.video_id}_features.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"✅ Features saved: {features_path.name}")
        
        # Run model inference
        logger.info("\nRunning model inference...")
        feature_cols = [
            'black_pixel_ratio', 'skin_pixel_ratio', 'teacher_presence',
            'global_ssim', 'global_mse', 'histogram_correlation',
            'edge_count', 'edge_change_rate',
            'zone_left_ssim', 'zone_left_edge_density',
            'zone_center_ssim', 'zone_center_edge_density',
            'zone_right_ssim', 'zone_right_edge_density',
            'ssim_rolling_mean', 'ssim_rolling_std', 'ssim_rolling_max',
            'edge_rolling_mean', 'edge_rolling_std', 'edge_rolling_max',
            'edge_decay_velocity'
        ]
        X = features_df[feature_cols].values
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1
        
        # Log probability statistics
        logger.info(f"\nProbability Statistics:")
        logger.info(f"  Min:  {probabilities.min():.4f}")
        logger.info(f"  Max:  {probabilities.max():.4f}")
        logger.info(f"  Mean: {probabilities.mean():.4f}")
        logger.info(f"  Median: {np.median(probabilities):.4f}")
        logger.info(f"  P95: {np.percentile(probabilities, 95):.4f}")
        logger.info(f"  P99: {np.percentile(probabilities, 99):.4f}")
        
        # Count how many would be detected at different thresholds
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            count = (probabilities >= thresh).sum()
            logger.info(f"  Threshold {thresh}: {count} frames ({count/len(probabilities)*100:.2f}%)")
        
        # Apply threshold (model trained on sparse data: only 11 transition frames)
        features_df['prediction'] = (probabilities >= self.prediction_threshold).astype(int)
        features_df['probability'] = probabilities
        
        # Temporal Smoothing (NMS) - Same as training pipeline
        logger.info(f"\nTemporal Smoothing (NMS)")
        logger.info(f"Window: {self.lookback_seconds} seconds")
        logger.info(f"Goal: Collapse multiple detections into single transitions")
        
        original_positives = (features_df['prediction'] == 1).sum()
        logger.info(f"Before smoothing: {original_positives} detections")
        
        # Find all predicted transitions
        transitions = features_df[features_df['prediction'] == 1].copy()
        
        if len(transitions) == 0:
            logger.warning("⚠️ No frames above threshold - using adaptive approach")
            # Use 99th percentile or frames with probability > mean + 2*std
            p99 = np.percentile(probabilities, 99)
            mean_prob = probabilities.mean()
            std_prob = probabilities.std()
            adaptive_threshold = max(p99, mean_prob + 2*std_prob)
            
            logger.info(f"  Adaptive threshold: {adaptive_threshold:.6f}")
            features_df['prediction'] = (probabilities >= adaptive_threshold).astype(int)
            transitions = features_df[features_df['prediction'] == 1].copy()
            logger.info(f"  Frames selected: {len(transitions)}")
        
        # Sort by timestamp
        transitions = transitions.sort_values('timestamp_seconds')
        
        # Cluster transitions within window
        clusters = []
        if len(transitions) > 0:
            current_cluster = [transitions.iloc[0]]
            
            for i in range(1, len(transitions)):
                current = transitions.iloc[i]
                prev = current_cluster[-1]
                
                # Check if within window
                time_diff = current['timestamp_seconds'] - prev['timestamp_seconds']
                if time_diff <= self.lookback_seconds:
                    current_cluster.append(current)
                else:
                    # Start new cluster
                    clusters.append(current_cluster)
                    current_cluster = [current]
            
            # Add last cluster
            if current_cluster:
                clusters.append(current_cluster)
        
        logger.info(f"Clusters formed: {len(clusters)}")
        
        # For each cluster, keep only the frame with highest probability
        kept_indices = set()
        for cluster in clusters:
            if len(cluster) == 1:
                kept_indices.add(cluster[0].name)
            else:
                # Find frame with max probability in cluster
                cluster_df = pd.DataFrame(cluster)
                max_prob_idx = cluster_df['probability'].idxmax()
                kept_indices.add(max_prob_idx)
        
        # Reset predictions: keep only high-confidence detections
        features_df['smoothed_prediction'] = 0
        features_df.loc[list(kept_indices), 'smoothed_prediction'] = 1
        
        smoothed_positives = (features_df['smoothed_prediction'] == 1).sum()
        logger.info(f"After smoothing: {smoothed_positives} detections")
        logger.info(f"Removed: {original_positives - smoothed_positives} false alarms")
        
        # Save predictions
        predictions_path = self.lecture_dir / f"{self.video_id}_predictions.csv"
        features_df.to_csv(predictions_path, index=False)
        logger.info(f"✅ Predictions saved: {predictions_path.name}")
        
        # Log statistics
        logger.info(f"\nTransition frames detected: {smoothed_positives}/{len(features_df)} ({smoothed_positives/len(features_df)*100:.2f}%)")
        
        return features_df, duration
    
    def deduplicate_transitions(self, features_df):
        """
        STEP 3: SSIM-Based Deduplication
        Sequential comparison with SSIM-based deduplication and blank slide detection
        Uses logic from deduplicate_transitions.py
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: DEDUPLICATION (SSIM-BASED)")
        logger.info("="*70)
        
        # Get positive predictions (already temporally smoothed)
        positive_df = features_df[features_df['smoothed_prediction'] == 1].copy()
        
        if len(positive_df) == 0:
            logger.warning("⚠️ No transitions detected!")
            return []
        
        # Sort by timestamp - these are already clustered by NMS
        timestamps = sorted(positive_df['timestamp_seconds'].values)
        logger.info(f"Smoothed transitions: {len(timestamps)}")
        
        # SSIM-based deduplication with sequential comparison
        video_path = self.video_path
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        deduplicated = []
        previous_frame = None
        previous_timestamp = 0.0
        
        logger.info("\nApplying SSIM deduplication...")
        for idx, ts in enumerate(tqdm(timestamps, desc="Deduplicating")):
            # Extract frame at transition timestamp
            frame_num = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not extract frame at {ts:.2f}s")
                continue
            
            # Get edge count for blank detection (normalized value from features)
            edge_count = features_df[features_df['timestamp_seconds'] == ts]['edge_count'].iloc[0]
            is_blank = edge_count < self.blank_edge_threshold
            
            # Calculate SSIM with previous accepted slide (not just previous frame)
            if previous_frame is None:
                # First slide - always accept
                ssim_score = 0.0
                accept = True
                logger.info(f"  [1] FIRST SLIDE at {ts:.2f}s")
            else:
                # Convert to grayscale for SSIM
                gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize if needed
                if gray1.shape != gray2.shape:
                    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
                
                # Create board-only mask (exclude teacher/skin regions)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 0], dtype=np.uint8)
                upper_skin = np.array([20, 150, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                board_mask = cv2.bitwise_not(skin_mask)
                
                # Apply mask to focus on board content only (ignore teacher)
                gray1_masked = cv2.bitwise_and(gray1, gray1, mask=board_mask)
                gray2_masked = cv2.bitwise_and(gray2, gray2, mask=board_mask)
                
                # Calculate SSIM on board-only regions
                ssim_score = ssim(gray1_masked, gray2_masked)
                
                # Calculate time gap
                time_gap = ts - previous_timestamp
                
                # Deduplication logic (board-only SSIM, ignores teacher movement):
                # - Rapid-fire: < 2s apart + SSIM < 0.85 → accept (different content appearing quickly)
                # - Normal: SSIM < 0.95 → accept (board content changed)
                if time_gap < 2.0 and ssim_score < 0.85:
                    accept = True
                    logger.info(f"  [{len(deduplicated)+1}] RAPID-FIRE at {ts:.2f}s (gap={time_gap:.2f}s, SSIM={ssim_score:.3f})")
                elif ssim_score < self.ssim_dedup_threshold:
                    accept = True
                    logger.info(f"  [{len(deduplicated)+1}] ACCEPTED at {ts:.2f}s (SSIM={ssim_score:.3f})")
                else:
                    accept = False
                    logger.info(f"  [{len(deduplicated)+1}] DUPLICATE at {ts:.2f}s (SSIM={ssim_score:.3f}) - SKIPPED")
            
            if accept:
                deduplicated.append({
                    'index': len(deduplicated) + 1,
                    'timestamp': round(ts, 2),
                    'timestamp_formatted': self._format_timestamp(ts),
                    'is_blank': bool(is_blank),
                    'edge_count': round(float(edge_count), 3),
                    'ssim_from_prev': round(float(ssim_score), 3),
                    'audio_window': {
                        'start': round(previous_timestamp, 2),
                        'end': round(ts, 2),
                        'duration': round(ts - previous_timestamp, 2)
                    },
                    'preview_image': f"transition_previews/transition_{len(deduplicated)+1:03d}.jpg",
                    'notes': ''
                })
                
                # Update previous frame and timestamp to the accepted slide
                previous_frame = frame.copy()
                previous_timestamp = ts
                
                if is_blank:
                    logger.warning(f"    ⚠️ BLANK SLIDE detected (edge={edge_count:.3f})")
        
        cap.release()
        
        logger.info(f"\n✅ Deduplicated transitions: {len(deduplicated)}")
        blank_count = sum(1 for t in deduplicated if t['is_blank'])
        logger.info(f"   Blank slides: {blank_count}")
        logger.info(f"   Content slides: {len(deduplicated) - blank_count}")
        
        # Save transitions.json
        transitions_data = {
            'video_id': self.video_id,
            'generated_at': datetime.now().isoformat(),
            'total_transitions': len(deduplicated),
            'blank_transitions': blank_count,
            'parameters': {
                'prediction_threshold': self.prediction_threshold,
                'ssim_dedup_threshold': self.ssim_dedup_threshold,
                'blank_edge_threshold': self.blank_edge_threshold,
                'rapid_fire_window': 2.0,
                'rapid_fire_ssim': 0.85
            },
            'transitions': deduplicated
        }
        
        transitions_path = self.lecture_dir / 'transitions.json'
        with open(transitions_path, 'w') as f:
            json.dump(transitions_data, f, indent=2)
        logger.info(f"✅ Saved: {transitions_path.name}")
        
        # Extract preview images
        self._extract_previews(deduplicated)
        
        return deduplicated
    
    def _extract_previews(self, transitions):
        """Extract 640px preview images for transitions"""
        logger.info("\nExtracting preview images...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        preview_dir = self.lecture_dir / 'transition_previews'
        
        for trans in tqdm(transitions, desc="Previews"):
            ts = trans['timestamp']
            frame_num = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Resize to 640px width
                height, width = frame.shape[:2]
                new_width = 640
                new_height = int(height * (new_width / width))
                resized = cv2.resize(frame, (new_width, new_height))
                
                preview_path = preview_dir / f"transition_{trans['index']:03d}.jpg"
                cv2.imwrite(str(preview_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        cap.release()
        logger.info(f"✅ Preview images saved to: {preview_dir.name}/")
    
    def extract_best_slides(self, transitions):
        """
        STEP 4: Best Slide Extraction
        Using enhanced content-first approach with adaptive window
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: BEST SLIDE EXTRACTION (CONTENT-FIRST)")
        logger.info("="*70)
        
        cap = cv2.VideoCapture(str(self.video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        slides_dir = self.lecture_dir / 'slides'
        metadata_entries = []
        
        pbar = tqdm(transitions, desc=f"Extracting {self.video_id}")
        
        for idx, trans in enumerate(pbar):
            try:
                t_curr = trans['timestamp']
                t_prev = transitions[idx-1]['timestamp'] if idx > 0 else 0.0
                
                # Find best frame in adaptive window
                best_frame, best_timestamp, scoring_details = self._find_best_frame(
                    self.video_path, t_curr, t_prev, video_fps
                )
                
                if best_frame is not None:
                    # Save slide
                    slide_filename = f"{self.video_id}_slide_{trans['index']:03d}.png"
                    slide_path = slides_dir / slide_filename
                    cv2.imwrite(str(slide_path), best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    
                    # Metadata
                    metadata_entry = {
                        'slide_number': trans['index'],
                        'filename': slide_filename,
                        'capture_timestamp': round(best_timestamp, 2),
                        'transition_timestamp': t_curr,
                        'audio_start_time': trans['audio_window']['start'],
                        'audio_end_time': trans['audio_window']['end'],
                        'transition_was_blank': trans['is_blank'],
                        'scoring': scoring_details
                    }
                    metadata_entries.append(metadata_entry)
                    
                    pbar.set_postfix({
                        'score': f"{scoring_details['final_score']:.3f}",
                        'quality': f"{scoring_details['quality_score']:.3f}",
                        'sharp': f"{scoring_details['sharpness_score']:.2f}",
                        'edges': scoring_details['edge_count']
                    })
                
            except Exception as e:
                logger.error(f"Error extracting slide {trans['index']}: {e}")
        
        pbar.close()
        
        # Save metadata
        metadata = {
            'video_id': self.video_id,
            'extraction_method': 'content_first_enhanced_v2',
            'extraction_params': {
                'lookback_seconds': self.lookback_seconds,
                'sampling_fps': 10,
                'scoring_formula': 'Quality × TeacherPenalty × (1 + Proximity × 0.15)',
                'quality_formula': '(Sharpness × Brightness × ∛(Board × Edges × Distribution))',
                'teacher_penalty': '1 / (1 + teacher_presence × 5)',
                'optimization': '10 FPS sampling for 3x speedup'
            },
            'slides': metadata_entries
        }
        
        metadata_path = self.lecture_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Extracted {len(metadata_entries)} slides to: {slides_dir.name}/")
        logger.info(f"✅ Metadata saved: {metadata_path.name}")
    
    def _find_best_frame(self, video_path, t_curr, t_prev, fps):
        """
        Find best frame in adaptive window between t_prev and t_curr.
        Uses enhanced content-first scoring from extract_best_slides.py
        
        Adaptive Window with Transition Buffer:
        - Start: Max of (T_curr - lookback_seconds) OR (T_prev + 1s)
        - End: T_curr - 1.0s (buffer to avoid fade-out/transition effects)
        
        The 1-second buffer ensures we capture slides BEFORE transition effects 
        (fade-out, wipe, blur) begin, avoiding unclear/fading content.
        
        Returns: (best_frame, best_timestamp, scoring_details)
        """
        # Adaptive window with transition buffer
        window_start = max(t_curr - self.lookback_seconds, t_prev + 1.0)
        window_end = t_curr - 1.0  # 1s buffer before transition
        
        if window_start >= window_end:
            window_start = max(0, t_curr - 2.0)
            window_end = max(window_start + 1.0, t_curr - 0.5)
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Handle video open failure
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            # Return frame at t_curr as fallback
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_MSEC, t_curr * 1000)
            ret, fallback_frame = cap.read()
            cap.release()
            return fallback_frame, t_curr, {'final_score': 0.0, 'fallback': True}
        
        start_frame = int(window_start * fps)
        end_frame = int(window_end * fps)
        
        # SPEED OPTIMIZATION: Sample at 10 FPS instead of 30 FPS for 3x speedup
        # For 30fps video: samples every 3rd frame (10fps)
        # This reduces processing from ~300 frames to ~100 frames per slide
        sampling_step = max(1, int(fps / 10))
        
        best_score = -1
        best_frame = None
        best_timestamp = t_curr
        best_details = {}
        
        for frame_num in range(start_frame, end_frame + 1, sampling_step):
            frame_time = frame_num / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            score, details = self._score_frame(frame, frame_time, window_start, window_end)
            
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_timestamp = frame_time
                best_details = details.copy()
                best_details['window_start'] = window_start
                best_details['window_end'] = window_end
                best_details['frames_evaluated'] = end_frame - start_frame + 1
        
        cap.release()
        
        # Fallback if no frame was found
        if best_frame is None:
            logger.warning(f"No valid frame found in window [{window_start:.2f}s - {window_end:.2f}s], using t_curr={t_curr:.2f}s")
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
                'teacher_presence': 0.0,
                'teacher_penalty': 1.0,
                'proximity_bias': 1.0,
                'proximity_multiplier': 1.0,
                'window_start': window_start,
                'window_end': window_end,
                'frames_evaluated': 0,
                'fallback': True
            }
        
        return best_frame, best_timestamp, best_details
    
    def _score_frame(self, frame, frame_time, window_start, window_end):
        """
        Score a single frame using enhanced content-first approach with teacher occlusion penalty.
        Returns: (final_score, scoring_details)
        
        Enhanced Formula:
        Content = ∛(Board × Edges × Distribution)
        Quality = Content × Sharpness × Brightness
        TeacherPenalty = 1 / (1 + teacher_presence × 5)
        Final = Quality × TeacherPenalty × (1 + Proximity × 0.15)
        """
        # 1. Detect board visibility
        board_visibility_score = self._detect_board_visibility(frame)
        
        # 2. Calculate edge score with distribution
        edge_score_normalized, edge_count, distribution_score = self._calculate_edge_score(frame)
        
        # 3. Calculate sharpness (motion blur detection)
        sharpness_score = self._calculate_sharpness(frame)
        
        # 4. Calculate brightness quality
        brightness_quality = self._calculate_brightness_quality(frame)
        
        # 5. Detect teacher presence (OCCLUSION PENALTY)
        teacher_presence_score = self._detect_teacher_presence(frame)
        
        # 6. Calculate proximity bias (reduced to 15%)
        if window_end == window_start:
            proximity_bias = 1.0
        else:
            proximity_bias = (frame_time - window_start) / (window_end - window_start)
        
        # 7. Content quality: requires board + text + distribution
        # Using geometric mean (cube root) to require all three
        content_score = (board_visibility_score * edge_score_normalized * distribution_score) ** (1/3)
        
        # 8. Overall quality: content + sharpness + brightness
        quality_score = content_score * sharpness_score * brightness_quality
        
        # 9. Teacher occlusion penalty: heavily penalize frames with teacher in the way
        # Formula: 1 / (1 + teacher × 5)
        # Examples: 0% teacher = 1.0x (no penalty), 20% teacher = 0.5x (50% penalty)
        teacher_penalty = 1.0 / (1.0 + (teacher_presence_score * 5.0))
        
        # 10. Final score with teacher penalty and proximity bias
        proximity_multiplier = 1.0 + (proximity_bias * 0.15)
        final_score = quality_score * teacher_penalty * proximity_multiplier
        
        # Scoring details for debugging and metadata
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
            'teacher_presence': teacher_presence_score,
            'teacher_penalty': teacher_penalty,
            'proximity_bias': proximity_bias,
            'proximity_multiplier': proximity_multiplier
        }
        
        return final_score, scoring_details
    
    def _detect_board_visibility(self, frame):
        """
        Detect board/chalk visibility using HSV color masking
        Returns: Board visibility score (0.0 = no board, 1.0 = full board visible)
        """
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
    
    def _calculate_edge_score(self, frame):
        """
        Calculate edge score using Canny + distribution quality.
        Higher score = more text visible and well-distributed.
        
        Returns: (normalized_edge_score, raw_edge_count, distribution_score)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
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
        if len(region_densities) > 0 and np.mean(region_densities) > 0:
            std_dev = np.std(region_densities)
            mean_density = np.mean(region_densities)
            # Coefficient of variation (normalized)
            cv = std_dev / (mean_density + 1e-6)
            distribution_score = max(1.0 - (cv / 2.0), 0.3)  # Penalize high variation
        else:
            distribution_score = 0.5
        
        return edge_score_normalized, edge_count, distribution_score
    
    def _calculate_sharpness(self, frame):
        """
        Calculate sharpness using Laplacian variance to detect motion blur.
        Higher score = sharper image (less blur from teacher/camera movement).
        
        Returns: normalized sharpness score (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize: typical sharp lecture frames have variance 100-1000
        normalized = min(laplacian_var / 200.0, 1.0)
        return normalized
    
    def _calculate_brightness_quality(self, frame):
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
    
    def _detect_teacher_presence(self, frame):
        """
        Detect teacher/person presence using skin color detection.
        Higher score = more teacher visible (blocking the board).
        
        Returns: skin pixel ratio (0.0 = no teacher, 1.0 = full frame)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        total_pixels = height * width
        
        # Detect skin tones (both light and dark)
        # Light skin range
        lower_skin_light = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin_light = np.array([20, 150, 255], dtype=np.uint8)
        mask_light = cv2.inRange(hsv, lower_skin_light, upper_skin_light)
        
        # Dark skin range
        lower_skin_dark = np.array([0, 20, 0], dtype=np.uint8)
        upper_skin_dark = np.array([20, 150, 120], dtype=np.uint8)
        mask_dark = cv2.inRange(hsv, lower_skin_dark, upper_skin_dark)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask_light, mask_dark)
        
        # Calculate skin pixel ratio
        skin_pixels = cv2.countNonZero(skin_mask)
        skin_ratio = skin_pixels / total_pixels
        
        return skin_ratio
    
    def generate_readme(self, transitions, duration):
        """Generate README.md summary"""
        logger.info("\nGenerating README.md...")
        
        blank_count = sum(1 for t in transitions if t['is_blank'])
        content_count = len(transitions) - blank_count
        
        readme_content = f"""# {self.video_id}

## Video Information
- **Duration**: {self._format_timestamp(duration)} ({duration:.2f}s)
- **Total Transitions Detected**: {len(transitions)}
- **Content Slides**: {content_count}
- **Blank Slides**: {blank_count}
- **Processed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Details
- **Model**: {self.model_path.name}
- **Prediction Threshold**: {self.prediction_threshold}
- **SSIM Dedup Threshold**: {self.ssim_dedup_threshold}
- **F1-Score**: 98.68% (on test set)

## Output Structure
```
{self.video_id}/
├── audio/                    # Extracted audio WAV
├── slides/                   # Best quality slide PNGs
├── transition_previews/      # Preview JPEGs (640px)
├── transitions.json          # Deduplicated timestamps + metadata
├── metadata.json             # Slide extraction details
└── README.md                 # This file
```

## Usage

### Extract Audio Segment
```python
import json
with open('transitions.json') as f:
    data = json.load(f)
    
# Get audio for slide 1
slide_1 = data['transitions'][0]
start = slide_1['audio_window']['start']
end = slide_1['audio_window']['end']

# Use ffmpeg
ffmpeg -i audio/{self.video_id}.wav -ss {{start}} -to {{end}} slide_1_audio.wav
```

### View Slide with Timestamp
```python
# Load metadata
with open('metadata.json') as f:
    meta = json.load(f)
    
for slide in meta['slides']:
    print(f"Slide {{slide['slide_number']}}: {{slide['filename']}}")
    print(f"  Captured at: {{slide['capture_timestamp']}}s")
    print(f"  Quality Score: {{slide['scoring']['quality_score']:.3f}}")
```

## Slide Quality Metrics
Each slide is scored on:
- **Board Visibility** (blackboard/whiteboard/greenboard detection)
- **Edge Density** (text/chalk clarity)
- **Distribution** (text spread across frame)
- **Sharpness** (motion blur detection)
- **Brightness** (exposure quality)
- **Proximity** (position in adaptive window)

## Next Steps
1. Review slides in `slides/` folder
2. Extract audio segments for each slide
3. Use multimodal AI (GPT-4 Vision/Gemini) for note generation
4. Combine slide images + audio + AI notes into final document
"""
        
        readme_path = self.lecture_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"✅ README saved: {readme_path.name}")
    
    def _format_timestamp(self, seconds):
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def process(self):
        """Execute complete pipeline"""
        logger.info("\n" + "="*70)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Step 1: Audio extraction
        self.extract_audio()
        
        # Step 2: Feature extraction & prediction
        features_df, duration = self.extract_features_and_predict()
        
        # Step 3: Deduplication
        transitions = self.deduplicate_transitions(features_df)
        
        if len(transitions) == 0:
            logger.error("❌ No transitions found. Pipeline aborted.")
            return
        
        # Step 4: Best slide extraction
        self.extract_best_slides(transitions)
        
        # Step 5: Generate README
        self.generate_readme(transitions, duration)
        
        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Output directory: {self.lecture_dir}")
        logger.info(f"Slides extracted: {len(transitions)}")
        logger.info("\nNext steps:")
        logger.info("  1. Review slides in slides/ folder")
        logger.info("  2. Extract audio segments using audio_window from transitions.json")
        logger.info("  3. Use multimodal AI for note generation")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Smart Notes Generator - Process new lecture video end-to-end',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_new_lecture.py path/to/lecture.mp4
  python process_new_lecture.py lecture.mp4 --model models/my_model.joblib
  python process_new_lecture.py lecture.mp4 --threshold 0.10 --ssim 0.90
  python process_new_lecture.py lecture.mp4 --feature-fps 5 --slide-fps 30
        """
    )
    
    parser.add_argument('video_path', help='Path to lecture video file')
    parser.add_argument('--output-dir', default='data/lectures', help='Output base directory')
    parser.add_argument('--model', default='models/xgboost_model_20260126_160645.pkl', help='Trained model path')
    parser.add_argument('--threshold', type=float, default=0.01, help='Prediction threshold (default: 0.01, model uses sparse training data)')
    parser.add_argument('--ssim', type=float, default=0.95, help='SSIM dedup threshold (default: 0.95, board-only comparison)')
    parser.add_argument('--blank-threshold', type=float, default=0.02, help='Blank edge threshold (default: 0.02 for HD videos)')
    parser.add_argument('--lookback', type=float, default=10.0, help='Adaptive window lookback (default: 10.0s)')
    parser.add_argument('--feature-fps', type=int, default=5, help='FPS for feature extraction (default: 5)')
    parser.add_argument('--slide-fps', type=int, default=30, help='FPS for slide extraction (default: 30)')
    
    args = parser.parse_args()
    
    try:
        processor = SmartNotesProcessor(
            video_path=args.video_path,
            output_base_dir=args.output_dir,
            model_path=args.model,
            prediction_threshold=args.threshold,
            ssim_dedup_threshold=args.ssim,
            blank_edge_threshold=args.blank_threshold,
            lookback_seconds=args.lookback,
            feature_extraction_fps=args.feature_fps,
            slide_extraction_fps=args.slide_fps
        )
        
        processor.process()
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
