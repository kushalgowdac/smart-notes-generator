"""
Generate HTML preview for final deduplicated slides
Creates visual galleries in data/final_slides folder
"""

import cv2
import json
import base64
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalSlidePreviewGenerator:
    """Generate HTML previews for final deduplicated slides"""
    
    def __init__(self, video_dir='data/videos', final_slides_json='data/transition_previews/deduplicated_transitions.json',
                 output_dir='data/transition_previews', thumbnail_width=320):
        self.video_dir = Path(video_dir)
        self.final_slides_json = Path(final_slides_json)
        self.output_dir = Path(output_dir)
        self.thumbnail_width = thumbnail_width
        
        logger.info("Initialized FinalSlidePreviewGenerator")
        logger.info(f"  Input JSON: {self.final_slides_json}")
        logger.info(f"  Output Dir: {self.output_dir}")
    
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
    
    def generate_video_preview(self, video_id, slides):
        """Generate HTML preview for a video's final slides"""
        logger.info(f"Generating preview for {video_id} ({len(slides)} transitions)")
        
        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
        
        # Extract thumbnails
        thumbnails = []
        for slide in slides:
            img_data = self.extract_thumbnail(video_path, slide['timestamp'])
            if img_data:
                thumbnails.append({
                    'slide_number': slide['slide_number'],
                    'timestamp': slide['timestamp'],
                    'time_formatted': self.format_time(slide['timestamp']),
                    'is_blank': slide['is_blank'],
                    'edge_count': slide['edge_count'],
                    'ssim_from_prev': slide['ssim_from_prev'],
                    'audio_duration': slide['audio_duration'],
                    'image_data': img_data
                })
        
        # Generate HTML
        html = self._create_html_template(video_id, thumbnails)
        
        # Create video-specific output directory
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save
        output_path = video_output_dir / f"{video_id}_deduplicated_transitions.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"  Saved: {output_path.name}")
        return output_path
    
    def _create_html_template(self, video_id, thumbnails):
        """Create HTML template"""
        blank_count = sum(1 for t in thumbnails if t['is_blank'])
        
        gallery_items = ""
        for thumb in thumbnails:
            blank_badge = " 🚫 BLANK" if thumb['is_blank'] else ""
            border_style = "border: 3px solid #ff9800;" if thumb['is_blank'] else ""
            
            gallery_items += f"""
        <div class="slide-card" style="{border_style}">
            <img src="data:image/jpeg;base64,{thumb['image_data']}" alt="Slide {thumb['slide_number']}">
            <div class="slide-info">
                <div class="slide-header">
                    <span class="slide-number">Slide #{thumb['slide_number']}{blank_badge}</span>
                    <span class="timestamp">{thumb['time_formatted']}</span>
                </div>
                <div class="slide-meta">
                    <div class="meta-item">
                        <span class="label">SSIM:</span>
                        <span class="value">{thumb['ssim_from_prev']:.3f}</span>
                    </div>
                    <div class="meta-item">
                        <span class="label">Edge:</span>
                        <span class="value">{thumb['edge_count']:.3f}</span>
                    </div>
                    <div class="meta-item">
                        <span class="label">Audio:</span>
                        <span class="value">{thumb['audio_duration']:.1f}s</span>
                    </div>
                </div>
            </div>
        </div>
"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deduplicated Transitions - {video_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        .stats {{
            display: flex;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .info-box h3 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        .info-box p {{
            color: #856404;
            line-height: 1.6;
            margin: 5px 0;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        .slide-card {{
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }}
        .slide-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        }}
        .slide-card img {{
            width: 100%;
            display: block;
        }}
        .slide-info {{
            padding: 20px;
        }}
        .slide-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .slide-number {{
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
        }}
        .timestamp {{
            font-size: 16px;
            color: #666;
            background: #f0f0f0;
            padding: 5px 12px;
            border-radius: 20px;
        }}
        .slide-meta {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .meta-item {{
            text-align: center;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .meta-item .label {{
            display: block;
            font-size: 11px;
            color: #666;
            margin-bottom: 3px;
        }}
        .meta-item .value {{
            display: block;
            font-size: 16px;
            font-weight: bold;
            color: #333;
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
        <h1>✨ Deduplicated Transitions: {video_id}</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Unique Transitions</div>
                <div class="stat-value">{len(thumbnails)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Content Slides</div>
                <div class="stat-value">{len(thumbnails) - blank_count}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Blank Slides</div>
                <div class="stat-value">{blank_count}</div>
            </div>
        </div>
    </div>
    
    <div class="info-box">
        <h3>ℹ️ About This Preview</h3>
        <p>• These transitions were deduplicated using SSIM-based sequential comparison (threshold: 0.92)</p>
        <p>• Blank slides (edge count < 0.1) are marked with 🚫 and orange border</p>
        <p>• SSIM values show visual similarity to previous slide (lower = more different)</p>
        <p>• Audio duration is the time from previous slide to current slide</p>
    </div>
    
    <div class="gallery">
        {gallery_items}
    </div>
    
    <div class="footer">
        <strong>Smart Notes Generator - Deduplicated Transitions Preview</strong><br>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        {len(thumbnails)} unique transitions after deduplication
    </div>
</body>
</html>"""
    
    def generate_all_previews(self):
        """Generate previews for all videos in final_unique_slides.json"""
        logger.info("\n" + "="*70)
        logger.info("GENERATING FINAL SLIDE PREVIEWS")
        logger.info("="*70)
        
        # Load final slides JSON
        with open(self.final_slides_json, 'r') as f:
            data = json.load(f)
        
        videos = data['videos']
        logger.info(f"Found {len(videos)} videos")
        
        generated = []
        for idx, (video_id, slides) in enumerate(videos.items(), 1):
            logger.info(f"\n[{idx}/{len(videos)}] {video_id}")
            html_path = self.generate_video_preview(video_id, slides)
            if html_path:
                generated.append(html_path)
        
        logger.info("\n" + "="*70)
        logger.info("PREVIEW GENERATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"Generated {len(generated)} HTML previews")
        logger.info(f"Location: {self.output_dir}")
        logger.info("\nAll previews organized in: data/transition_previews/<video_id>/")
        logger.info("  - <video_id>_transitions.html (raw model detections)")
        logger.info("  - <video_id>_deduplicated_transitions.html (final unique transitions)")
        logger.info("  - <video_id>_transitions.txt (timestamp list)")
        
        return generated


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HTML previews for final deduplicated slides')
    parser.add_argument('--video', type=str, help='Generate preview for single video')
    
    args = parser.parse_args()
    
    generator = FinalSlidePreviewGenerator()
    
    if args.video:
        # Single video
        with open(generator.final_slides_json, 'r') as f:
            data = json.load(f)
        
        if args.video in data['videos']:
            slides = data['videos'][args.video]
            generator.generate_video_preview(args.video, slides)
        else:
            logger.error(f"Video {args.video} not found in final_unique_slides.json")
    else:
        # All videos
        generator.generate_all_previews()


if __name__ == "__main__":
    main()
