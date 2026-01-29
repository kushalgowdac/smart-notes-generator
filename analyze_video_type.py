"""
Smart SSIM Threshold Recommender
Analyzes video slide patterns and recommends optimal deduplication threshold

Usage:
    python analyze_video_type.py data/lectures/my_lecture
    python analyze_video_type.py data/lectures/my_lecture --visualize
"""

import cv2
import json
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import argparse
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_frame_ssim(frame1, frame2):
    """Calculate SSIM between two frames"""
    if frame1 is None or frame2 is None:
        return 0.0
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # Board-focused SSIM (exclude teacher)
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 0], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv1, lower_skin, upper_skin)
    board_mask = cv2.bitwise_not(skin_mask)
    
    gray1_masked = cv2.bitwise_and(gray1, gray1, mask=board_mask)
    gray2_masked = cv2.bitwise_and(gray2, gray2, mask=board_mask)
    
    score = ssim(gray1_masked, gray2_masked)
    return score


def analyze_ssim_distribution(lecture_dir):
    """Analyze SSIM scores between consecutive slides"""
    lecture_path = Path(lecture_dir)
    transitions_file = lecture_path / 'transitions.json'
    slides_dir = lecture_path / 'slides'
    
    if not transitions_file.exists():
        logger.error(f"transitions.json not found in {lecture_dir}")
        return None
    
    if not slides_dir.exists():
        logger.error(f"slides/ directory not found in {lecture_dir}")
        return None
    
    with open(transitions_file, 'r') as f:
        data = json.load(f)
    
    video_id = data['video_id']
    transitions = data['transitions']
    
    logger.info(f"Loading {len(transitions)} slides...")
    
    # Calculate SSIM between consecutive slides
    ssim_scores = []
    previous_frame = None
    
    for trans in transitions:
        slide_file = slides_dir / f"{video_id}_slide_{trans['index']:03d}.png"
        
        if not slide_file.exists():
            continue
        
        current_frame = cv2.imread(str(slide_file))
        if current_frame is None:
            continue
        
        if previous_frame is not None:
            score = calculate_frame_ssim(current_frame, previous_frame)
            ssim_scores.append({
                'slide_from': trans['index'] - 1,
                'slide_to': trans['index'],
                'ssim': score
            })
        
        previous_frame = current_frame.copy()
    
    return ssim_scores


def detect_video_type(ssim_scores):
    """Classify video type based on SSIM distribution"""
    if not ssim_scores:
        return None
    
    scores = [s['ssim'] for s in ssim_scores]
    
    # Statistical analysis
    mean_ssim = np.mean(scores)
    median_ssim = np.median(scores)
    std_ssim = np.std(scores)
    
    # Count high-similarity pairs (likely duplicates)
    high_similarity_count = sum(1 for s in scores if s >= 0.85)
    high_similarity_ratio = high_similarity_count / len(scores)
    
    # Detect incremental vs discrete
    incremental_threshold = 0.80
    incremental_count = sum(1 for s in scores if s >= incremental_threshold)
    incremental_ratio = incremental_count / len(scores)
    
    # Classification
    if high_similarity_ratio > 0.3:
        video_type = "INCREMENTAL_WRITING"
        description = "Teacher gradually adds content to board (high duplicate risk)"
        recommended_ssim = 0.80  # Stricter
    elif high_similarity_ratio > 0.15:
        video_type = "MIXED"
        description = "Mix of slide changes and incremental updates"
        recommended_ssim = 0.85  # Balanced
    else:
        video_type = "DISCRETE_SLIDES"
        description = "Clear slide transitions with distinct content"
        recommended_ssim = 0.92  # Permissive
    
    return {
        'video_type': video_type,
        'description': description,
        'recommended_ssim': recommended_ssim,
        'stats': {
            'mean': round(mean_ssim, 3),
            'median': round(median_ssim, 3),
            'std': round(std_ssim, 3),
            'high_similarity_ratio': round(high_similarity_ratio, 3),
            'incremental_ratio': round(incremental_ratio, 3)
        }
    }


def predict_reduction(ssim_scores, threshold):
    """Predict how many slides would be removed at given threshold"""
    duplicates = sum(1 for s in ssim_scores if s['ssim'] >= threshold)
    total = len(ssim_scores) + 1  # +1 for first slide
    final = total - duplicates
    reduction_pct = (duplicates / total) * 100
    
    return {
        'original': total,
        'duplicates': duplicates,
        'final': final,
        'reduction_pct': round(reduction_pct, 1)
    }


def find_sample_duplicates(ssim_scores, threshold, limit=5):
    """Find sample slide pairs that would be removed"""
    candidates = [s for s in ssim_scores if s['ssim'] >= threshold]
    candidates.sort(key=lambda x: x['ssim'], reverse=True)
    return candidates[:limit]


def main():
    parser = argparse.ArgumentParser(
        description='Analyze video type and recommend SSIM threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_video_type.py data/lectures/deadlock_os
  python analyze_video_type.py data/lectures/chemistry_01_english --visualize
        """
    )
    
    parser.add_argument('lecture_dir', help='Path to lecture directory')
    parser.add_argument('--visualize', action='store_true', help='Show SSIM distribution plot')
    
    args = parser.parse_args()
    
    lecture_path = Path(args.lecture_dir)
    
    if not lecture_path.exists():
        logger.error(f"Lecture directory not found: {lecture_path}")
        return 1
    
    logger.info("="*70)
    logger.info(f"ANALYZING VIDEO TYPE: {lecture_path.name}")
    logger.info("="*70)
    logger.info("")
    
    # Analyze SSIM distribution
    ssim_scores = analyze_ssim_distribution(lecture_path)
    
    if ssim_scores is None or len(ssim_scores) == 0:
        logger.error("Failed to analyze SSIM distribution")
        return 1
    
    logger.info(f"✓ Analyzed {len(ssim_scores)} consecutive slide pairs")
    logger.info("")
    
    # Detect video type
    analysis = detect_video_type(ssim_scores)
    
    # Print results
    logger.info("="*70)
    logger.info("VIDEO TYPE ANALYSIS")
    logger.info("="*70)
    logger.info(f"Classification: {analysis['video_type']}")
    logger.info(f"Description: {analysis['description']}")
    logger.info("")
    
    logger.info("Statistics:")
    logger.info(f"  Mean SSIM: {analysis['stats']['mean']}")
    logger.info(f"  Median SSIM: {analysis['stats']['median']}")
    logger.info(f"  Std Dev: {analysis['stats']['std']}")
    logger.info(f"  High similarity ratio: {analysis['stats']['high_similarity_ratio']} (>0.85)")
    logger.info(f"  Incremental ratio: {analysis['stats']['incremental_ratio']} (>0.80)")
    logger.info("")
    
    # Recommended threshold
    logger.info("="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)
    logger.info(f"Recommended SSIM threshold: {analysis['recommended_ssim']}")
    logger.info("")
    
    # Impact prediction at different thresholds
    logger.info("Impact prediction at different thresholds:")
    logger.info("")
    logger.info("Threshold | Original | Duplicates | Final | Reduction")
    logger.info("-"*60)
    
    for threshold in [0.80, 0.85, 0.90, 0.92, 0.95]:
        pred = predict_reduction(ssim_scores, threshold)
        logger.info(f"  {threshold}    |    {pred['original']:2d}    |     {pred['duplicates']:2d}     |  {pred['final']:2d}   |  {pred['reduction_pct']:4.1f}%")
    
    logger.info("")
    
    # Sample duplicates at recommended threshold
    samples = find_sample_duplicates(ssim_scores, analysis['recommended_ssim'])
    
    if samples:
        logger.info(f"Sample duplicates at SSIM={analysis['recommended_ssim']}:")
        for i, sample in enumerate(samples, 1):
            logger.info(f"  {i}. Slide {sample['slide_from']} → {sample['slide_to']}: SSIM={sample['ssim']:.3f}")
        logger.info("")
    
    # Usage instruction
    logger.info("="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info(f"To apply recommended threshold, run:")
    logger.info(f"  python rerun_deduplication.py {lecture_path} --ssim {analysis['recommended_ssim']}")
    logger.info("")
    
    # Visualization
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            
            scores = [s['ssim'] for s in ssim_scores]
            
            plt.figure(figsize=(12, 6))
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
            plt.axvline(analysis['recommended_ssim'], color='red', linestyle='--', 
                       label=f"Recommended: {analysis['recommended_ssim']}")
            plt.xlabel('SSIM Score')
            plt.ylabel('Frequency')
            plt.title(f'SSIM Distribution - {analysis["video_type"]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Sequential plot
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(scores)+1), scores, marker='o', markersize=3, alpha=0.6)
            plt.axhline(analysis['recommended_ssim'], color='red', linestyle='--',
                       label=f"Recommended: {analysis['recommended_ssim']}")
            plt.axhline(0.85, color='orange', linestyle=':', alpha=0.5, label='Duplicate zone (>0.85)')
            plt.xlabel('Slide Pair Index')
            plt.ylabel('SSIM Score')
            plt.title('SSIM Between Consecutive Slides')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_file = lecture_path / 'ssim_analysis.png'
            plt.savefig(output_file, dpi=150)
            logger.info(f"✓ Visualization saved: {output_file}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not installed - skipping visualization")
            logger.info("Install with: pip install matplotlib")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
