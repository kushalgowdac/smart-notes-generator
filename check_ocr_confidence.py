"""
OCR Confidence Checker
Analyze OCR quality and identify slides with low confidence text extraction

Usage:
    python check_ocr_confidence.py data/lectures/my_lecture
    python check_ocr_confidence.py data/lectures/my_lecture --threshold 0.7
    python check_ocr_confidence.py data/lectures/my_lecture --show-problems
"""

import json
import cv2
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_ocr_cache(lecture_dir, confidence_threshold=0.75):
    """Analyze OCR confidence from cached results"""
    
    lecture_path = Path(lecture_dir)
    ocr_cache = lecture_path / 'ocr_cache.json'
    
    if not ocr_cache.exists():
        logger.error(f"OCR cache not found: {ocr_cache}")
        logger.info("Run notes generation first to create OCR cache")
        return None
    
    # Load OCR cache
    with open(ocr_cache, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    # Analyze each slide
    results = []
    low_confidence_count = 0
    empty_text_count = 0
    
    for item in ocr_data:
        slide_name = item.get('image_name', 'unknown')
        ocr_text = item.get('ocr_text', '')
        confidence = item.get('confidence', 0.0)
        blocks = item.get('blocks', [])
        
        # Calculate stats
        text_length = len(ocr_text.strip())
        num_blocks = len(blocks)
        
        # Block-level confidence analysis
        if blocks:
            block_confidences = [b.get('confidence', 0.0) for b in blocks]
            min_conf = min(block_confidences) if block_confidences else 0.0
            max_conf = max(block_confidences) if block_confidences else 0.0
            avg_conf = sum(block_confidences) / len(block_confidences) if block_confidences else 0.0
        else:
            min_conf = max_conf = avg_conf = 0.0
        
        # Classify
        if text_length == 0:
            status = "EMPTY"
            empty_text_count += 1
        elif confidence < confidence_threshold:
            status = "LOW_CONFIDENCE"
            low_confidence_count += 1
        else:
            status = "OK"
        
        results.append({
            'slide_name': slide_name,
            'status': status,
            'confidence': confidence,
            'text_length': text_length,
            'num_blocks': num_blocks,
            'min_block_conf': min_conf,
            'max_block_conf': max_conf,
            'avg_block_conf': avg_conf,
            'ocr_text': ocr_text
        })
    
    return {
        'total_slides': len(results),
        'low_confidence': low_confidence_count,
        'empty_text': empty_text_count,
        'threshold': confidence_threshold,
        'slides': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Check OCR confidence and identify problem slides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_ocr_confidence.py data/lectures/deadlock_os
  python check_ocr_confidence.py data/lectures/chemistry_01 --threshold 0.8
  python check_ocr_confidence.py data/lectures/algo_1 --show-problems

The tool analyzes ocr_cache.json to find slides with:
  - Low OCR confidence (< threshold)
  - Empty/no text detected
  - Inconsistent block-level confidence
        """
    )
    
    parser.add_argument('lecture_dir', help='Path to lecture directory')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Confidence threshold (default: 0.75)')
    parser.add_argument('--show-problems', action='store_true',
                       help='Display problem slides visually')
    parser.add_argument('--export-report', action='store_true',
                       help='Save detailed JSON report')
    
    args = parser.parse_args()
    
    lecture_path = Path(args.lecture_dir)
    
    if not lecture_path.exists():
        logger.error(f"Lecture directory not found: {lecture_path}")
        return 1
    
    logger.info("="*70)
    logger.info(f"OCR CONFIDENCE CHECK: {lecture_path.name}")
    logger.info("="*70)
    logger.info(f"Confidence threshold: {args.threshold}")
    logger.info("")
    
    # Analyze
    analysis = analyze_ocr_cache(lecture_path, args.threshold)
    
    if not analysis:
        return 1
    
    # Summary
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Total slides: {analysis['total_slides']}")
    logger.info(f"Low confidence: {analysis['low_confidence']} ({100*analysis['low_confidence']/analysis['total_slides']:.1f}%)")
    logger.info(f"Empty text: {analysis['empty_text']} ({100*analysis['empty_text']/analysis['total_slides']:.1f}%)")
    logger.info("")
    
    # Problem slides
    problem_slides = [s for s in analysis['slides'] if s['status'] != 'OK']
    
    if problem_slides:
        logger.warning(f"⚠️  {len(problem_slides)} slides need attention:")
        logger.info("")
        logger.info("Slide Name                          | Status          | Conf  | Text Len")
        logger.info("-"*75)
        
        for slide in problem_slides:
            logger.info(f"{slide['slide_name']:35s} | {slide['status']:15s} | {slide['confidence']:.3f} | {slide['text_length']:4d}")
        
        logger.info("")
        logger.info("⚠️  Recommendations:")
        if analysis['low_confidence'] > 0:
            logger.info("   - Low confidence slides may have blurry/poor quality text")
            logger.info("   - Consider re-extracting with higher quality frames")
            logger.info("   - Manual review recommended before using notes")
        if analysis['empty_text'] > 0:
            logger.info("   - Empty slides might be blank/transition frames")
            logger.info("   - Or slides with only images/diagrams (no text)")
            logger.info("   - Review manually to confirm")
        logger.info("")
        
        # Show problem slides
        if args.show_problems:
            slides_dir = lecture_path / 'slides'
            
            if not slides_dir.exists():
                logger.warning("slides/ directory not found, cannot display images")
            else:
                logger.info("Press any key to view next problem slide, 'q' to quit")
                
                for slide in problem_slides:
                    img_path = slides_dir / slide['slide_name']
                    
                    if not img_path.exists():
                        continue
                    
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        continue
                    
                    # Add overlay
                    h, w = img.shape[:2]
                    overlay = img.copy()
                    cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                    
                    cv2.putText(img, f"Slide: {slide['slide_name']}", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(img, f"Status: {slide['status']}", (20, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, f"Confidence: {slide['confidence']:.3f}", (20, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(img, f"Text length: {slide['text_length']} chars", (20, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(img, f"OCR blocks: {slide['num_blocks']}", (20, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow(f"OCR Confidence Check - {lecture_path.name}", img)
                    key = cv2.waitKey(0)
                    
                    if key == ord('q') or key == 27:
                        break
                
                cv2.destroyAllWindows()
    
    else:
        logger.info("✓ All slides have acceptable OCR confidence!")
        logger.info("  Ready for notes generation")
    
    # Export report
    if args.export_report:
        report_file = lecture_path / 'ocr_confidence_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Detailed report saved: {report_file}")
    
    return 0 if len(problem_slides) == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
