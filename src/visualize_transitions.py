"""
Transition Visualization Tool
Generates HTML galleries to review all detected transitions for each video
"""

import os
import cv2
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransitionVisualizer:
    """Generate visual galleries of all transitions for review"""
    
    def __init__(self, video_dir='data/videos', predictions_csv='data/all_predictions.csv',
                 output_dir='data/transition_previews', thumbnail_width=320):
        self.video_dir = Path(video_dir)
        self.predictions_csv = Path(predictions_csv)
        self.output_dir = Path(output_dir)
        self.thumbnail_width = thumbnail_width
        
        # Create main output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Initialized TransitionVisualizer")
        logger.info(f"  Video Directory: {self.video_dir}")
        logger.info(f"  Predictions CSV: {self.predictions_csv}")
        logger.info(f"  Output Directory: {self.output_dir}")
        logger.info(f"  Thumbnail Width: {self.thumbnail_width}px")
    
    def extract_transition_thumbnails(self, video_path, timestamps):
        """
        Extract thumbnail images at transition timestamps
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamp values in seconds
            
        Returns:
            List of dictionaries with timestamp and base64 image data
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        thumbnails = []
        
        for timestamp in timestamps:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Resize to thumbnail
                height = int(frame.shape[0] * (self.thumbnail_width / frame.shape[1]))
                thumbnail = cv2.resize(frame, (self.thumbnail_width, height))
                
                # Convert to base64 for HTML embedding
                _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                thumbnails.append({
                    'timestamp': timestamp,
                    'image_data': img_base64,
                    'time_formatted': self._format_time(timestamp)
                })
            else:
                logger.warning(f"Could not extract frame at {timestamp:.2f}s")
        
        cap.release()
        return thumbnails
    
    def _format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def generate_html_gallery(self, video_id, thumbnails, metadata, video_output_dir):
        """
        Generate HTML gallery for a video's transitions
        
        Args:
            video_id: Video identifier
            thumbnails: List of thumbnail dictionaries
            metadata: Dictionary with statistics
            video_output_dir: Path to video-specific output directory
        """
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transitions - {video_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .stats {{
            display: flex;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .stat {{
            display: flex;
            flex-direction: column;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            color: #333;
            font-size: 24px;
            font-weight: bold;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .transition-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .transition-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        
        .transition-card img {{
            width: 100%;
            display: block;
            border-bottom: 1px solid #eee;
        }}
        
        .transition-info {{
            padding: 15px;
        }}
        
        .transition-time {{
            font-size: 18px;
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 5px;
        }}
        
        .transition-index {{
            color: #666;
            font-size: 14px;
        }}
        
        .footer {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #666;
        }}
        
        .legend {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .legend h3 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        
        .legend p {{
            color: #856404;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Transitions Preview: {video_id}</h1>
        <div class="stats">
            <div class="stat">
                <span class="stat-label">Total Transitions</span>
                <span class="stat-value">{total_transitions}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Video Duration</span>
                <span class="stat-value">{duration}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Generated</span>
                <span class="stat-value">{generated_at}</span>
            </div>
        </div>
    </div>
    
    <div class="legend">
        <h3>ℹ️ How to Review</h3>
        <p>
            • Each card shows a frame captured at the predicted transition timestamp<br>
            • Look for: Empty slides, duplicate content, or non-transition frames<br>
            • Blue timestamp shows exact position in video (MM:SS format)<br>
            • Hover over cards for better visibility
        </p>
    </div>
    
    <div class="gallery">
        {gallery_items}
    </div>
    
    <div class="footer">
        Generated by Smart Notes Generator - Transition Visualizer<br>
        {total_transitions} transitions detected in {video_id}
    </div>
</body>
</html>
"""
        
        gallery_items = ""
        for idx, thumb in enumerate(thumbnails, 1):
            gallery_items += f"""
        <div class="transition-card">
            <img src="data:image/jpeg;base64,{thumb['image_data']}" alt="Transition {idx}">
            <div class="transition-info">
                <div class="transition-time">{thumb['time_formatted']}</div>
                <div class="transition-index">Transition #{idx} • {thumb['timestamp']:.2f}s</div>
            </div>
        </div>
"""
        
        html_content = html_template.format(
            video_id=video_id,
            total_transitions=metadata['total_transitions'],
            duration=metadata['duration'],
            generated_at=metadata['generated_at'],
            gallery_items=gallery_items
        )
        
        # Save HTML file in video-specific folder
        output_path = video_output_dir / f"{video_id}_transitions.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML: {output_path.name}")
        return output_path
    
    def generate_transitions_txt(self, video_id, transition_times, video_output_dir):
        """
        Generate simple text file with transition timestamps
        
        Args:
            video_id: Video identifier
            transition_times: List of transition timestamps in seconds
            video_output_dir: Path to video-specific output directory
        """
        output_path = video_output_dir / f"{video_id}_transitions.txt"
        
        with open(output_path, 'w') as f:
            f.write(f"# Transitions for {video_id}\n")
            f.write(f"# Total: {len(transition_times)} transitions\n")
            f.write(f"# Format: MM:SS (seconds)\n")
            f.write("#" + "="*50 + "\n\n")
            
            for idx, timestamp in enumerate(transition_times, 1):
                time_formatted = self._format_time(timestamp)
                f.write(f"{idx:3d}. {time_formatted} ({timestamp:.2f}s)\n")
        
        logger.info(f"Saved TXT: {output_path.name}")
        return output_path
    
    def visualize_video(self, video_id):
        """
        Generate transition preview gallery for a specific video
        
        Args:
            video_id: Video identifier
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Visualizing Transitions: {video_id}")
        logger.info(f"{'='*70}")
        
        # Create video-specific output directory
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load predictions
        predictions_df = pd.read_csv(self.predictions_csv)
        video_df = predictions_df[predictions_df['video_id'] == video_id].copy()
        
        if len(video_df) == 0:
            logger.error(f"No predictions found for {video_id}")
            return None
        
        # Get transitions (label=1)
        transitions = video_df[video_df['label'] == 1].copy()
        transition_times = sorted(transitions['timestamp_seconds'].unique())
        
        logger.info(f"Found {len(transition_times)} transitions")
        
        # Find video file
        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        cap.release()
        
        # Extract thumbnails
        logger.info("Extracting transition thumbnails...")
        thumbnails = self.extract_transition_thumbnails(video_path, transition_times)
        
        # Generate HTML gallery
        metadata = {
            'total_transitions': len(transition_times),
            'duration': self._format_time(duration_seconds),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        
        html_path = self.generate_html_gallery(video_id, thumbnails, metadata, video_output_dir)
        
        # Generate transitions TXT file
        txt_path = self.generate_transitions_txt(video_id, transition_times, video_output_dir)
        
        logger.info(f"\n✅ Gallery created!")
        logger.info(f"   Folder: {video_output_dir}")
        logger.info(f"   HTML: {html_path.name}")
        logger.info(f"   TXT: {txt_path.name}")
        logger.info(f"   Transitions: {len(transition_times)}")
        
        return video_output_dir
    
    def visualize_all_videos(self):
        """Generate galleries for all videos in predictions CSV"""
        logger.info("\n" + "="*70)
        logger.info("Generating Transition Galleries for All Videos")
        logger.info("="*70)
        
        # Load predictions
        predictions_df = pd.read_csv(self.predictions_csv)
        video_ids = sorted(predictions_df['video_id'].unique())
        
        logger.info(f"Found {len(video_ids)} videos")
        
        generated_files = []
        
        for idx, video_id in enumerate(video_ids, 1):
            logger.info(f"\n[{idx}/{len(video_ids)}] Processing: {video_id}")
            video_dir = self.visualize_video(video_id)
            if video_dir:
                generated_files.append(video_dir)
        
        logger.info("\n" + "="*70)
        logger.info("ALL GALLERIES GENERATED!")
        logger.info("="*70)
        logger.info(f"Total videos: {len(generated_files)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("\nEach video folder contains:")
        logger.info("  - <video_id>_transitions.html (visual gallery)")
        logger.info("  - <video_id>_transitions.txt (timestamp list)")
        logger.info(f"\nTo review: Open HTML files in {self.output_dir}/<video_id>/")
        
        return generated_files


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Visualize detected transitions for video review',
        epilog='''
Examples:
  # Visualize single video
  python src/visualize_transitions.py --video toc_1
  
  # Visualize all videos
  python src/visualize_transitions.py --all
  
  # Custom thumbnail size
  python src/visualize_transitions.py --video algo_1 --thumbnail-width 480
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, help='Process single video by ID')
    parser.add_argument('--all', action='store_true', help='Process all videos')
    parser.add_argument('--predictions', type=str, default='data/all_predictions.csv',
                       help='Path to predictions CSV')
    parser.add_argument('--thumbnail-width', type=int, default=320,
                       help='Thumbnail width in pixels (default: 320)')
    parser.add_argument('--list-videos', action='store_true',
                       help='List all available videos')
    
    args = parser.parse_args()
    
    visualizer = TransitionVisualizer(
        predictions_csv=args.predictions,
        thumbnail_width=args.thumbnail_width
    )
    
    if args.list_videos:
        # List available videos
        df = pd.read_csv(args.predictions)
        video_ids = sorted(df['video_id'].unique())
        
        print("\nAvailable videos in predictions CSV:")
        print("-" * 50)
        for idx, video_id in enumerate(video_ids, 1):
            video_df = df[df['video_id'] == video_id]
            transitions = len(video_df[video_df['label'] == 1])
            print(f"{idx:3d}. {video_id:30s} ({transitions:4d} transitions)")
        print(f"\nTotal: {len(video_ids)} videos")
        
    elif args.video:
        # Single video
        visualizer.visualize_video(args.video)
        
    elif args.all:
        # All videos
        visualizer.visualize_all_videos()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
