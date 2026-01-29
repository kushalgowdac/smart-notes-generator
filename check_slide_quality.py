"""
Slide Quality Checker
Detects blurry, overexposed, or teacher-occluded slides before notes generation

Usage:
    python check_slide_quality.py data/lectures/my_lecture
    python check_slide_quality.py data/lectures/my_lecture --threshold 80
    python check_slide_quality.py data/lectures/my_lecture --show-bad-slides
"""

import cv2
import json
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_sharpness(image):
    """Calculate Laplacian variance (sharpness)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_brightness(image):
    """Calculate mean brightness (0-255)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def calculate_teacher_presence(image):
    """Calculate percentage of frame with teacher (skin tones)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Skin color detection
    lower_skin = np.array([0, 20, 0], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    teacher_pixels = np.sum(skin_mask > 0)
    total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
    
    teacher_ratio = teacher_pixels / total_pixels
    return teacher_ratio * 100  # Percentage


def calculate_contrast(image):
    """Calculate RMS contrast"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rms_contrast = np.std(gray)
    return rms_contrast


def analyze_slide_quality(slide_path):
    """Analyze single slide and return quality metrics"""
    image = cv2.imread(str(slide_path))
    
    if image is None:
        return None
    
    metrics = {
        'sharpness': round(calculate_sharpness(image), 2),
        'brightness': round(calculate_brightness(image), 2),
        'teacher_presence': round(calculate_teacher_presence(image), 2),
        'contrast': round(calculate_contrast(image), 2)
    }
    
    return metrics


def classify_quality(metrics, thresholds):
    """Classify slide quality and identify issues"""
    issues = []
    quality_score = 100  # Start at perfect
    
    # Sharpness check (Laplacian variance)
    if metrics['sharpness'] < thresholds['sharpness_min']:
        issues.append(f"BLURRY (sharpness={metrics['sharpness']:.1f} < {thresholds['sharpness_min']})")
        quality_score -= 30
    
    # Brightness check
    if metrics['brightness'] < thresholds['brightness_min']:
        issues.append(f"UNDEREXPOSED (brightness={metrics['brightness']:.1f} < {thresholds['brightness_min']})")
        quality_score -= 20
    elif metrics['brightness'] > thresholds['brightness_max']:
        issues.append(f"OVEREXPOSED (brightness={metrics['brightness']:.1f} > {thresholds['brightness_max']})")
        quality_score -= 20
    
    # Teacher occlusion check
    if metrics['teacher_presence'] > thresholds['teacher_max']:
        issues.append(f"TEACHER_OCCLUDED (coverage={metrics['teacher_presence']:.1f}% > {thresholds['teacher_max']}%)")
        quality_score -= 25
    
    # Contrast check
    if metrics['contrast'] < thresholds['contrast_min']:
        issues.append(f"LOW_CONTRAST (contrast={metrics['contrast']:.1f} < {thresholds['contrast_min']})")
        quality_score -= 15
    
    # Overall classification
    if quality_score >= 90:
        classification = "EXCELLENT"
    elif quality_score >= 70:
        classification = "GOOD"
    elif quality_score >= 50:
        classification = "ACCEPTABLE"
    else:
        classification = "POOR"
    
    return {
        'classification': classification,
        'quality_score': max(0, quality_score),
        'issues': issues
    }


def main():
    parser = argparse.ArgumentParser(
        description='Check slide quality before notes generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_slide_quality.py data/lectures/deadlock_os
  python check_slide_quality.py data/lectures/deadlock_os --threshold 70
  python check_slide_quality.py data/lectures/deadlock_os --show-bad-slides
        """
    )
    
    parser.add_argument('lecture_dir', help='Path to lecture directory')
    parser.add_argument('--threshold', type=int, default=70, 
                       help='Quality score threshold for warnings (default: 70)')
    parser.add_argument('--show-bad-slides', action='store_true', 
                       help='Display poor quality slides in window')
    
    args = parser.parse_args()
    
    lecture_path = Path(args.lecture_dir)
    
    if not lecture_path.exists():
        logger.error(f"Lecture directory not found: {lecture_path}")
        return 1
    
    transitions_file = lecture_path / 'transitions.json'
    slides_dir = lecture_path / 'slides'
    
    if not transitions_file.exists():
        logger.error(f"transitions.json not found")
        return 1
    
    if not slides_dir.exists():
        logger.error(f"slides/ directory not found")
        return 1
    
    with open(transitions_file, 'r') as f:
        data = json.load(f)
    
    video_id = data['video_id']
    transitions = data['transitions']
    
    logger.info("="*70)
    logger.info(f"SLIDE QUALITY CHECK: {lecture_path.name}")
    logger.info("="*70)
    logger.info(f"Total slides: {len(transitions)}")
    logger.info("")
    
    # Quality thresholds (tuned for lecture videos)
    thresholds = {
        'sharpness_min': 100,      # Laplacian variance
        'brightness_min': 50,       # Too dark
        'brightness_max': 220,      # Too bright
        'teacher_max': 15,          # Teacher occlusion percentage
        'contrast_min': 30          # RMS contrast
    }
    
    # Analyze all slides
    results = []
    poor_count = 0
    acceptable_count = 0
    good_count = 0
    excellent_count = 0
    
    logger.info("Analyzing slides...")
    for trans in transitions:
        slide_file = slides_dir / f"{video_id}_slide_{trans['index']:03d}.png"
        
        if not slide_file.exists():
            logger.warning(f"Slide {trans['index']:03d} not found: {slide_file.name}")
            continue
        
        metrics = analyze_slide_quality(slide_file)
        if metrics is None:
            continue
        
        quality = classify_quality(metrics, thresholds)
        
        result = {
            'index': trans['index'],
            'timestamp': trans['timestamp'],
            'metrics': metrics,
            'quality': quality,
            'file': slide_file
        }
        
        results.append(result)
        
        # Count classifications
        if quality['classification'] == 'EXCELLENT':
            excellent_count += 1
        elif quality['classification'] == 'GOOD':
            good_count += 1
        elif quality['classification'] == 'ACCEPTABLE':
            acceptable_count += 1
        else:
            poor_count += 1
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("QUALITY SUMMARY")
    logger.info("="*70)
    logger.info(f"Excellent: {excellent_count} ({100*excellent_count/len(results):.1f}%)")
    logger.info(f"Good: {good_count} ({100*good_count/len(results):.1f}%)")
    logger.info(f"Acceptable: {acceptable_count} ({100*acceptable_count/len(results):.1f}%)")
    logger.info(f"Poor: {poor_count} ({100*poor_count/len(results):.1f}%)")
    logger.info("")
    
    # Flag problematic slides
    flagged = [r for r in results if r['quality']['quality_score'] < args.threshold]
    
    if flagged:
        logger.warning(f"⚠️  {len(flagged)} slides below quality threshold ({args.threshold}):")
        logger.info("")
        logger.info("Slide | Time   | Score | Issues")
        logger.info("-"*70)
        
        for result in flagged:
            issues_str = ", ".join(result['quality']['issues']) if result['quality']['issues'] else "None"
            logger.info(f" {result['index']:3d}  | {result['timestamp']:6.1f}s | {result['quality']['quality_score']:3d}   | {issues_str}")
        
        logger.info("")
        logger.info("⚠️  WARNING: These slides may produce poor quality notes")
        logger.info("   Consider re-extracting with different parameters or manual review")
        logger.info("")
        
        # Show bad slides
        if args.show_bad_slides and flagged:
            logger.info("Press any key to view next slide, 'q' to quit")
            for result in flagged:
                img = cv2.imread(str(result['file']))
                
                # Add quality info overlay
                h, w = img.shape[:2]
                overlay = img.copy()
                cv2.rectangle(overlay, (10, 10), (600, 150), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                
                cv2.putText(img, f"Slide {result['index']} @ {result['timestamp']:.1f}s", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(img, f"Quality: {result['quality']['classification']} ({result['quality']['quality_score']})", 
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if result['quality']['issues']:
                    cv2.putText(img, f"Issues: {result['quality']['issues'][0]}", 
                               (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow(f"Poor Quality Slides - {lecture_path.name}", img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
            
            cv2.destroyAllWindows()
    
    else:
        logger.info(f"✓ All slides meet quality threshold ({args.threshold})")
        logger.info("  Ready for notes generation!")
    
    # Save report
    report_file = lecture_path / 'quality_report.json'
    with open(report_file, 'w') as f:
        json.dump({
            'video_id': video_id,
            'total_slides': len(results),
            'thresholds': thresholds,
            'summary': {
                'excellent': excellent_count,
                'good': good_count,
                'acceptable': acceptable_count,
                'poor': poor_count
            },
            'flagged_count': len(flagged),
            'slides': results
        }, f, indent=2)
    
    logger.info(f"✓ Detailed report saved: {report_file}")
    logger.info("")
    
    return 0 if poor_count == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
