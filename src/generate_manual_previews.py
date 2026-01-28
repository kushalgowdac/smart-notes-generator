"""
Generate HTML previews for MANUAL (ground truth) transitions
Creates both raw and deduplicated versions for comparison with model predictions
"""

import cv2
import json
import base64
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ManualTransitionPreviewGenerator:
    """Generate previews for manual/ground truth transitions"""
    
    def __init__(self, video_dir='data/videos', ground_truth_dir='data/ground_truth',
                 master_dataset='data/master_dataset.csv', output_dir='data/transition_previews',
                 thumbnail_width=320, ssim_dedup_threshold=0.92):
        
        self.video_dir = Path(video_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.master_dataset = Path(master_dataset)
        self.output_dir = Path(output_dir)
        self.thumbnail_width = thumbnail_width
        self.ssim_dedup_threshold = ssim_dedup_threshold
        
        logger.info("Initialized ManualTransitionPreviewGenerator")
        logger.info(f"  Ground Truth Dir: {self.ground_truth_dir}")
        logger.info(f"  Master Dataset: {self.master_dataset}")
    
    def load_manual_transitions_raw(self, video_id):
        """Load raw manual transitions from master_dataset.csv (all label=1 frames)"""
        df = pd.read_csv(self.master_dataset)
        video_df = df[df['video_id'] == video_id].copy()
        
        # Get all frames labeled as 1 (transition)
        transitions = video_df[video_df['label'] == 1].copy()
        timestamps = sorted(transitions['timestamp_seconds'].unique())
        
        return timestamps
    
    def load_manual_transitions_deduplicated(self, video_id):
        """Load deduplicated manual transitions from transitions.txt"""
        transitions_file = self.ground_truth_dir / video_id / 'transitions.txt'
        
        if not transitions_file.exists():
            logger.warning(f"No transitions.txt found for {video_id}")
            return []
        
        timestamps = []
        with open(transitions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        # Parse timestamp (format: seconds as float)
                        timestamp = float(line)
                        timestamps.append(timestamp)
                    except ValueError:
                        continue
        
        return sorted(timestamps)
    
    def extract_thumbnail(self, video_path, timestamp):
        """Extract thumbnail at timestamp"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            height = int(frame.shape[0] * (self.thumbnail_width / frame.shape[1]))
            thumbnail = cv2.resize(frame, (self.thumbnail_width, height))
            _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
        return None
    
    def format_time(self, seconds):
        """Convert seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def generate_html(self, video_id, timestamps, output_path, is_deduplicated=False):
        """Generate HTML gallery for manual transitions"""
        logger.info(f"Generating {'deduplicated' if is_deduplicated else 'raw'} manual preview for {video_id}")
        
        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
        
        # Extract thumbnails
        thumbnails = []
        for idx, ts in enumerate(timestamps, 1):
            img_data = self.extract_thumbnail(video_path, ts)
            if img_data:
                thumbnails.append({
                    'number': idx,
                    'timestamp': ts,
                    'time_formatted': self.format_time(ts),
                    'image_data': img_data
                })
        
        # Generate HTML
        transition_type = "Deduplicated Manual" if is_deduplicated else "Raw Manual (11-frame labeled)"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{transition_type} Transitions - {video_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 32px;
        }}
        .badge {{
            display: inline-block;
            background: #11998e;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .stats {{
            display: flex;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 15px 25px;
            border-radius: 10px;
            color: white;
        }}
        .stat-label {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .info-box {{
            background: #d4edda;
            border: 2px solid #28a745;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .info-box h3 {{
            color: #155724;
            margin-bottom: 10px;
        }}
        .info-box p {{
            color: #155724;
            line-height: 1.6;
            margin: 5px 0;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        .transition-card {{
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            border: 3px solid #11998e;
        }}
        .transition-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        }}
        .transition-card img {{
            width: 100%;
            display: block;
        }}
        .transition-info {{
            padding: 20px;
        }}
        .transition-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .transition-number {{
            font-size: 18px;
            font-weight: bold;
            color: #11998e;
        }}
        .timestamp {{
            font-size: 16px;
            color: #666;
            background: #f0f0f0;
            padding: 5px 12px;
            border-radius: 20px;
        }}
        .footer {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: #666;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 {transition_type} Transitions: {video_id}</h1>
        <div class="badge">GROUND TRUTH DATA</div>
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Total Transitions</div>
                <div class="stat-value">{len(thumbnails)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Source</div>
                <div class="stat-value">{'transitions.txt' if is_deduplicated else 'master_dataset.csv'}</div>
            </div>
        </div>
    </div>
    
    <div class="info-box">
        <h3>ℹ️ About This Preview</h3>
        <p>• This shows <strong>MANUAL/GROUND TRUTH</strong> transitions (not model predictions)</p>
        <p>• {'Source: transitions.txt (human-labeled unique transitions)' if is_deduplicated else 'Source: master_dataset.csv (all frames labeled as 1, including 11 consecutive frames)'}</p>
        <p>• Use this to compare with model predictions and evaluate accuracy</p>
    </div>
    
    <div class="gallery">
"""
        
        for thumb in thumbnails:
            html += f"""
        <div class="transition-card">
            <img src="data:image/jpeg;base64,{thumb['image_data']}" alt="Transition {thumb['number']}">
            <div class="transition-info">
                <div class="transition-header">
                    <span class="transition-number">Manual #{thumb['number']}</span>
                    <span class="timestamp">{thumb['time_formatted']}</span>
                </div>
            </div>
        </div>
"""
        
        html += f"""
    </div>
    
    <div class="footer">
        <strong>Smart Notes Generator - Manual Transitions Preview</strong><br>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        {len(thumbnails)} ground truth transitions
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"  Saved: {output_path.name}")
        return output_path
    
    def generate_video_previews(self, video_id):
        """Generate both raw and deduplicated previews for a video"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Manual Transitions: {video_id}")
        logger.info(f"{'='*70}")
        
        # Create video folder
        video_folder = self.output_dir / video_id
        video_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate raw preview (from master_dataset.csv)
        raw_timestamps = self.load_manual_transitions_raw(video_id)
        logger.info(f"Raw manual transitions (label=1): {len(raw_timestamps)}")
        
        if raw_timestamps:
            raw_output = video_folder / f"{video_id}_manual_raw_transitions.html"
            self.generate_html(video_id, raw_timestamps, raw_output, is_deduplicated=False)
        
        # Generate deduplicated preview (from transitions.txt)
        dedup_timestamps = self.load_manual_transitions_deduplicated(video_id)
        logger.info(f"Deduplicated manual transitions (transitions.txt): {len(dedup_timestamps)}")
        
        if dedup_timestamps:
            dedup_output = video_folder / f"{video_id}_manual_deduplicated_transitions.html"
            self.generate_html(video_id, dedup_timestamps, dedup_output, is_deduplicated=True)
        
        return {
            'raw_count': len(raw_timestamps),
            'dedup_count': len(dedup_timestamps)
        }
    
    def generate_all_videos(self):
        """Generate manual previews for all videos"""
        logger.info("\n" + "="*70)
        logger.info("GENERATING MANUAL TRANSITION PREVIEWS")
        logger.info("="*70)
        
        # Get all videos from ground truth directory
        video_folders = [f for f in self.ground_truth_dir.iterdir() if f.is_dir()]
        
        results = {}
        for idx, video_folder in enumerate(sorted(video_folders), 1):
            video_id = video_folder.name
            logger.info(f"\n[{idx}/{len(video_folders)}]")
            
            stats = self.generate_video_previews(video_id)
            results[video_id] = stats
        
        logger.info("\n" + "="*70)
        logger.info("MANUAL PREVIEW GENERATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"Processed {len(results)} videos")
        
        # Summary
        for video_id, stats in sorted(results.items()):
            logger.info(f"  {video_id}: {stats['raw_count']} raw, {stats['dedup_count']} deduplicated")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HTML previews for manual/ground truth transitions')
    parser.add_argument('--video', type=str, help='Process single video')
    parser.add_argument('--all', action='store_true', help='Process all videos')
    
    args = parser.parse_args()
    
    generator = ManualTransitionPreviewGenerator()
    
    if args.video:
        generator.generate_video_previews(args.video)
    elif args.all:
        generator.generate_all_videos()
    else:
        logger.error("Please specify --video <id> or --all")


if __name__ == "__main__":
    main()
