"""
Re-run SSIM deduplication on already extracted slides
No video processing needed - works with existing slides and transitions.json
"""
import cv2
import json
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_frame_ssim(frame1, frame2):
    """Calculate SSIM between two frames (board-content focused)"""
    if frame1 is None or frame2 is None:
        return 0.0
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Resize to same dimensions if needed
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # Create mask to focus on board area (ignore teacher regions)
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    # Exclude skin tones
    lower_skin = np.array([0, 20, 0], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv1, lower_skin, upper_skin)
    
    # Invert to get board-only mask
    board_mask = cv2.bitwise_not(skin_mask)
    
    # Apply mask
    gray1_masked = cv2.bitwise_and(gray1, gray1, mask=board_mask)
    gray2_masked = cv2.bitwise_and(gray2, gray2, mask=board_mask)
    
    # Calculate SSIM on masked regions
    score = ssim(gray1_masked, gray2_masked)
    return score


def rerun_deduplication(lecture_dir, new_ssim_threshold=0.85):
    """Re-run deduplication with new SSIM threshold"""
    lecture_path = Path(lecture_dir)
    
    if not lecture_path.exists():
        logger.error(f"Lecture directory not found: {lecture_dir}")
        return
    
    # Load existing transitions.json
    transitions_file = lecture_path / 'transitions.json'
    if not transitions_file.exists():
        logger.error(f"transitions.json not found in {lecture_dir}")
        return
    
    with open(transitions_file, 'r') as f:
        data = json.load(f)
    
    existing_transitions = data['transitions']
    old_ssim = data['parameters']['ssim_dedup_threshold']
    
    logger.info("="*70)
    logger.info(f"RE-RUNNING DEDUPLICATION: {lecture_path.name}")
    logger.info("="*70)
    logger.info(f"Existing slides: {len(existing_transitions)}")
    logger.info(f"Old SSIM threshold: {old_ssim}")
    logger.info(f"New SSIM threshold: {new_ssim_threshold}")
    logger.info("")
    
    # Load slide images
    slides_dir = lecture_path / 'slides'
    if not slides_dir.exists():
        logger.error(f"Slides directory not found: {slides_dir}")
        return
    
    # Sequential comparison with new SSIM threshold
    final_transitions = []
    previous_frame = None
    previous_timestamp = 0.0
    skipped_count = 0
    
    for idx, trans in enumerate(existing_transitions, 1):
        # Load current slide image (use video_id from transitions.json, not folder name)
        slide_file = slides_dir / f"{data['video_id']}_slide_{trans['index']:03d}.png"
        
        if not slide_file.exists():
            logger.warning(f"Slide image not found: {slide_file.name}, keeping transition")
            final_transitions.append(trans)
            continue
        
        current_frame = cv2.imread(str(slide_file))
        
        if current_frame is None:
            logger.warning(f"Could not load {slide_file.name}, keeping transition")
            final_transitions.append(trans)
            continue
        
        # First slide - always accept
        if previous_frame is None:
            final_transitions.append(trans)
            previous_frame = current_frame.copy()
            previous_timestamp = trans['timestamp']
            logger.info(f"  [{idx}] KEPT at {trans['timestamp']}s (first slide)")
            continue
        
        # Calculate SSIM with previous accepted slide
        ssim_score = calculate_frame_ssim(current_frame, previous_frame)
        
        # Check if duplicate
        if ssim_score >= new_ssim_threshold:
            # KEEP LATEST: Replace previous slide with this newer version
            # (incremental updates - newer slide has more complete content)
            if len(final_transitions) > 0:
                old_timestamp = final_transitions[-1]['timestamp']
                # Remove the older incomplete slide
                final_transitions.pop()
                
                # Add current (newer) slide with updated audio window
                trans['audio_window']['start'] = previous_timestamp if len(final_transitions) > 0 else 0.0
                if len(final_transitions) > 0:
                    previous_timestamp = final_transitions[-1]['timestamp']
                else:
                    previous_timestamp = 0.0
                trans['audio_window']['start'] = previous_timestamp
                trans['audio_window']['duration'] = trans['timestamp'] - previous_timestamp
                trans['ssim_from_prev'] = round(ssim_score, 3)
                
                final_transitions.append(trans)
                previous_frame = current_frame.copy()
                previous_timestamp = trans['timestamp']
                
                logger.info(f"  [{idx}] REPLACED slide at {old_timestamp}s with newer at {trans['timestamp']}s (SSIM={ssim_score:.3f})")
            else:
                # Should not happen, but keep current if no previous
                final_transitions.append(trans)
                previous_frame = current_frame.copy()
                previous_timestamp = trans['timestamp']
                logger.info(f"  [{idx}] KEPT at {trans['timestamp']}s (SSIM={ssim_score:.3f})")
        else:
            # Different content - keep this slide
            trans['audio_window']['start'] = previous_timestamp
            trans['audio_window']['duration'] = trans['timestamp'] - previous_timestamp
            trans['ssim_from_prev'] = round(ssim_score, 3)
            
            final_transitions.append(trans)
            previous_frame = current_frame.copy()
            previous_timestamp = trans['timestamp']
            logger.info(f"  [{idx}] KEPT at {trans['timestamp']}s (SSIM={ssim_score:.3f})")
    
    # Renumber slides
    for i, trans in enumerate(final_transitions, 1):
        trans['index'] = i
    
    # Create new transitions.json
    new_data = {
        'video_id': data['video_id'],
        'generated_at': datetime.now().isoformat(),
        'total_transitions': len(final_transitions),
        'blank_transitions': sum(1 for t in final_transitions if t.get('is_blank', False)),
        'parameters': {
            **data['parameters'],
            'ssim_dedup_threshold': new_ssim_threshold,
            'reprocessed': True,
            'original_count': len(existing_transitions)
        },
        'transitions': final_transitions
    }
    
    # Save to NEW file (don't overwrite original)
    new_file = lecture_path / f'transitions_reprocessed_ssim{new_ssim_threshold}.json'
    with open(new_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    logger.info(f"\n✓ Saved reprocessed transitions to: {new_file.name}")
    logger.info(f"✓ Original transitions.json NOT modified")
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Original slides: {len(existing_transitions)}")
    logger.info(f"Replaced with newer versions: {len(existing_transitions) - len(final_transitions)}")
    logger.info(f"Final slides: {len(final_transitions)}")
    logger.info(f"Reduction: {len(existing_transitions) - len(final_transitions)}/{len(existing_transitions)} ({100*(len(existing_transitions) - len(final_transitions))/len(existing_transitions):.1f}%)")
    logger.info("")
    
    if len(existing_transitions) != len(final_transitions):
        logger.info("Strategy: KEEP LATEST - Older incomplete slides replaced with newer versions")
        logger.info("Note: Slide images NOT deleted (for safety)")
        logger.info("      Only transitions.json was updated")


def main():
    parser = argparse.ArgumentParser(
        description='Re-run SSIM deduplication on already extracted slides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rerun_deduplication.py data/lectures/deadlock_os --ssim 0.85
  python rerun_deduplication.py data/lectures/deadlock_os --ssim 0.80
        """
    )
    
    parser.add_argument('lecture_dir', help='Path to lecture directory with slides/')
    parser.add_argument('--ssim', type=float, default=0.85, help='New SSIM threshold (default: 0.85)')
    
    args = parser.parse_args()
    
    rerun_deduplication(args.lecture_dir, args.ssim)


if __name__ == '__main__':
    main()
