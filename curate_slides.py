"""
Interactive Slide Curator
Manual review and curation of detected slides before notes generation

Usage:
    python curate_slides.py data/lectures/my_lecture
    python curate_slides.py data/lectures/my_lecture --auto-remove-duplicates
"""

import cv2
import json
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SlideCurator:
    """Interactive slide curation interface"""
    
    def __init__(self, lecture_dir):
        self.lecture_path = Path(lecture_dir)
        
        if not self.lecture_path.exists():
            raise FileNotFoundError(f"Lecture directory not found: {lecture_dir}")
        
        self.transitions_file = self.lecture_path / 'transitions.json'
        self.slides_dir = self.lecture_path / 'slides'
        
        if not self.transitions_file.exists():
            raise FileNotFoundError(f"transitions.json not found")
        
        if not self.slides_dir.exists():
            raise FileNotFoundError(f"slides/ directory not found")
        
        # Load transitions
        with open(self.transitions_file, 'r') as f:
            self.data = json.load(f)
        
        self.video_id = self.data['video_id']
        self.transitions = self.data['transitions']
        self.total_slides = len(self.transitions)
        
        # Curation state
        self.keep_flags = [True] * self.total_slides  # All kept by default
        self.current_index = 0
        self.changes_made = False
    
    def load_slide_image(self, slide_index):
        """Load slide image by index"""
        trans = self.transitions[slide_index]
        slide_file = self.slides_dir / f"{self.video_id}_slide_{trans['index']:03d}.png"
        
        if not slide_file.exists():
            logger.warning(f"Slide image not found: {slide_file.name}")
            return None
        
        return cv2.imread(str(slide_file))
    
    def create_info_panel(self, slide_index):
        """Create info panel overlay on image"""
        trans = self.transitions[slide_index]
        
        info_lines = [
            f"Slide {slide_index + 1}/{self.total_slides}",
            f"Original Index: {trans['index']}",
            f"Time: {trans['timestamp']:.1f}s",
            f"Status: {'KEEP' if self.keep_flags[slide_index] else 'SKIP'}",
            "",
            "Controls:",
            "  [Y] Keep    [N] Skip    [U] Undo",
            "  [←] Previous    [→] Next",
            "  [S] Save & Exit    [Q] Quit without save",
            "  [A] Auto-remove remaining duplicates",
            "",
            f"Kept: {sum(self.keep_flags)}/{self.total_slides}",
            f"Skipped: {self.total_slides - sum(self.keep_flags)}"
        ]
        
        return info_lines
    
    def display_slide(self, slide_index):
        """Display slide with info overlay"""
        img = self.load_slide_image(slide_index)
        
        if img is None:
            # Show placeholder
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "IMAGE NOT FOUND", (400, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Create overlay panel
        h, w = img.shape[:2]
        panel_height = 400
        panel_width = 450
        
        # Semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        
        # Add info text
        info_lines = self.create_info_panel(slide_index)
        y_offset = 40
        
        for i, line in enumerate(info_lines):
            if line.startswith("Slide "):
                # Title in cyan
                cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 0), 2)
            elif "Status:" in line:
                # Status with color
                status = "KEEP" if self.keep_flags[slide_index] else "SKIP"
                color = (0, 255, 0) if self.keep_flags[slide_index] else (0, 0, 255)
                cv2.putText(img, f"Status: {status}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            elif line.startswith("  ["):
                # Controls in yellow
                cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 255), 1)
            elif line.startswith("Kept:") or line.startswith("Skipped:"):
                # Stats in white
                cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1)
            else:
                # Regular text
                cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (200, 200, 200), 1)
            
            y_offset += 25
        
        # Add skip indicator if flagged
        if not self.keep_flags[slide_index]:
            # Red X overlay
            cv2.line(img, (w-200, 50), (w-50, 200), (0, 0, 255), 10)
            cv2.line(img, (w-50, 50), (w-200, 200), (0, 0, 255), 10)
            cv2.putText(img, "SKIPPED", (w-190, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, (0, 0, 255), 3)
        
        return img
    
    def process_key(self, key):
        """Process keyboard input"""
        
        # Keep slide
        if key == ord('y') or key == ord('Y'):
            if not self.keep_flags[self.current_index]:
                self.keep_flags[self.current_index] = True
                self.changes_made = True
                logger.info(f"Slide {self.current_index + 1} marked as KEEP")
            self.current_index = min(self.current_index + 1, self.total_slides - 1)
        
        # Skip slide
        elif key == ord('n') or key == ord('N'):
            if self.keep_flags[self.current_index]:
                self.keep_flags[self.current_index] = False
                self.changes_made = True
                logger.info(f"Slide {self.current_index + 1} marked as SKIP")
            self.current_index = min(self.current_index + 1, self.total_slides - 1)
        
        # Undo (toggle current)
        elif key == ord('u') or key == ord('U'):
            self.keep_flags[self.current_index] = not self.keep_flags[self.current_index]
            self.changes_made = True
            status = "KEEP" if self.keep_flags[self.current_index] else "SKIP"
            logger.info(f"Slide {self.current_index + 1} toggled to {status}")
        
        # Next slide
        elif key == 83 or key == ord('d') or key == ord('D'):  # Right arrow or D
            self.current_index = min(self.current_index + 1, self.total_slides - 1)
        
        # Previous slide
        elif key == 81 or key == ord('a') or key == ord('A'):  # Left arrow or A
            self.current_index = max(self.current_index - 1, 0)
        
        # Auto-remove duplicates
        elif key == ord('r') or key == ord('R'):
            removed = self.auto_remove_duplicates()
            logger.info(f"Auto-removed {removed} duplicate slides")
            self.changes_made = True
        
        # Save and exit
        elif key == ord('s') or key == ord('S'):
            return 'save'
        
        # Quit without save
        elif key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            return 'quit'
        
        return 'continue'
    
    def auto_remove_duplicates(self):
        """Auto-detect and remove obvious duplicates using SSIM"""
        from skimage.metrics import structural_similarity as ssim
        
        removed_count = 0
        previous_img = None
        
        for i in range(self.total_slides):
            if not self.keep_flags[i]:
                continue  # Already skipped
            
            current_img = self.load_slide_image(i)
            if current_img is None:
                continue
            
            if previous_img is not None:
                # Calculate SSIM
                gray1 = cv2.cvtColor(previous_img, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                
                if gray1.shape == gray2.shape:
                    ssim_score = ssim(gray1, gray2)
                    
                    # High similarity = duplicate
                    if ssim_score >= 0.90:
                        self.keep_flags[i] = False
                        removed_count += 1
                        logger.info(f"  Auto-removed slide {i+1} (SSIM={ssim_score:.3f})")
            
            if self.keep_flags[i]:
                previous_img = current_img.copy()
        
        return removed_count
    
    def save_curated_transitions(self):
        """Save curated transitions.json"""
        
        # Filter kept transitions
        curated_transitions = [
            trans for i, trans in enumerate(self.transitions)
            if self.keep_flags[i]
        ]
        
        # Renumber slides
        for i, trans in enumerate(curated_transitions, 1):
            trans['index'] = i
        
        # Create new data
        curated_data = {
            'video_id': self.data['video_id'],
            'generated_at': datetime.now().isoformat(),
            'total_transitions': len(curated_transitions),
            'blank_transitions': sum(1 for t in curated_transitions if t.get('is_blank', False)),
            'parameters': {
                **self.data.get('parameters', {}),
                'manually_curated': True,
                'original_count': self.total_slides,
                'curated_at': datetime.now().isoformat()
            },
            'transitions': curated_transitions
        }
        
        # Backup original
        backup_file = self.lecture_path / f'transitions_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(backup_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"✓ Backed up original: {backup_file.name}")
        
        # Save curated version
        with open(self.transitions_file, 'w') as f:
            json.dump(curated_data, f, indent=2)
        logger.info(f"✓ Saved curated transitions.json")
        
        return len(curated_transitions)
    
    def run(self):
        """Run interactive curation"""
        
        logger.info("="*70)
        logger.info(f"INTERACTIVE SLIDE CURATOR: {self.lecture_path.name}")
        logger.info("="*70)
        logger.info(f"Total slides: {self.total_slides}")
        logger.info("")
        logger.info("Keyboard controls:")
        logger.info("  [Y] Keep    [N] Skip    [U] Undo")
        logger.info("  [←/→] or [A/D] Navigate    [R] Auto-remove duplicates")
        logger.info("  [S] Save & Exit    [Q] Quit without save")
        logger.info("")
        
        window_name = f"Slide Curator - {self.lecture_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        while True:
            # Display current slide
            display_img = self.display_slide(self.current_index)
            cv2.imshow(window_name, display_img)
            
            # Wait for key
            key = cv2.waitKey(0) & 0xFF
            
            # Process key
            action = self.process_key(key)
            
            if action == 'save':
                cv2.destroyAllWindows()
                
                if self.changes_made:
                    logger.info("")
                    logger.info("="*70)
                    logger.info("SAVING CHANGES")
                    logger.info("="*70)
                    
                    final_count = self.save_curated_transitions()
                    
                    logger.info("")
                    logger.info("Summary:")
                    logger.info(f"  Original slides: {self.total_slides}")
                    logger.info(f"  Kept: {final_count}")
                    logger.info(f"  Removed: {self.total_slides - final_count}")
                    logger.info(f"  Reduction: {100 * (self.total_slides - final_count) / self.total_slides:.1f}%")
                    logger.info("")
                    logger.info("✓ Curation complete!")
                else:
                    logger.info("")
                    logger.info("No changes made, transitions.json not modified")
                
                return True
            
            elif action == 'quit':
                cv2.destroyAllWindows()
                
                if self.changes_made:
                    logger.warning("")
                    logger.warning("Changes discarded (not saved)")
                
                logger.info("Curation cancelled")
                return False


def main():
    parser = argparse.ArgumentParser(
        description='Interactive slide curation tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python curate_slides.py data/lectures/deadlock_os
  python curate_slides.py data/lectures/chemistry_01_english

Keyboard Controls:
  Y - Keep current slide and move to next
  N - Skip current slide and move to next
  U - Undo (toggle keep/skip status)
  ← → or A D - Navigate between slides
  R - Auto-remove remaining duplicates (SSIM > 0.90)
  S - Save changes and exit
  Q or ESC - Quit without saving
        """
    )
    
    parser.add_argument('lecture_dir', help='Path to lecture directory')
    
    args = parser.parse_args()
    
    try:
        curator = SlideCurator(args.lecture_dir)
        success = curator.run()
        
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
