"""
Organize Lecture Folders - Create Intelligent Structure
Creates per-video lecture folders with:
- transitions.json (deduplicated timestamps including blanks)
- transition_previews/ (images of all transitions including blanks)
- slides/ (future: best quality slides)
- audio/ (future: extracted audio segments)
- metadata.json (future: best slide details)
"""

import json
import cv2
from pathlib import Path
import base64
import logging
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LectureFolderOrganizer:
    """Organize lecture content into intelligent folder structure"""
    
    def __init__(self, 
                 videos_dir: str = 'data/videos',
                 deduplicated_json: str = 'data/transition_previews/deduplicated_transitions.json',
                 output_base: str = 'data/lectures'):
        
        self.videos_dir = Path(videos_dir)
        self.deduplicated_json = Path(deduplicated_json)
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized LectureFolderOrganizer")
        logger.info(f"Output directory: {self.output_base}")
    
    def create_folder_structure(self, video_id: str) -> Dict[str, Path]:
        """Create complete folder structure for a video"""
        base = self.output_base / video_id
        
        folders = {
            'base': base,
            'transition_previews': base / 'transition_previews',
            'slides': base / 'slides',
            'audio': base / 'audio',
        }
        
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        return folders
    
    def extract_transition_frame(self, video_path: Path, timestamp: float, 
                                 output_path: Path, max_width: int = 640) -> bool:
        """Extract a single frame at timestamp"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False
            
            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.warning(f"Failed to extract frame at {timestamp}s")
                return False
            
            # Resize if needed
            height, width = frame.shape[:2]
            if width > max_width:
                ratio = max_width / width
                new_width = max_width
                new_height = int(height * ratio)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Save frame
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return True
            
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return False
    
    def create_transitions_json(self, video_id: str, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create transitions.json for a single video"""
        
        # Sort by timestamp
        transitions_sorted = sorted(transitions, key=lambda x: x['timestamp'])
        
        # Create clean structure
        transitions_data = {
            'video_id': video_id,
            'generated_at': datetime.now().isoformat(),
            'total_transitions': len(transitions_sorted),
            'blank_count': sum(1 for t in transitions_sorted if t.get('is_blank', False)),
            'content_count': sum(1 for t in transitions_sorted if not t.get('is_blank', False)),
            'transitions': []
        }
        
        # Add each transition with relevant info
        for idx, trans in enumerate(transitions_sorted, 1):
            transition_entry = {
                'index': idx,
                'timestamp': trans['timestamp'],
                'timestamp_formatted': self._format_timestamp(trans['timestamp']),
                'is_blank': trans.get('is_blank', False),
                'ssim_score': trans.get('ssim_score', None),
                'edge_count': trans.get('edge_count', None),
                'has_audio': trans.get('has_audio', False),
                'audio_window': {
                    'start': trans.get('audio_window_start', None),
                    'end': trans.get('audio_window_end', None),
                    'duration': trans.get('audio_duration', None)
                },
                'preview_image': f'transition_previews/transition_{idx:03d}.jpg',
                'notes': ''  # For future manual annotations
            }
            
            transitions_data['transitions'].append(transition_entry)
        
        return transitions_data
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def create_readme(self, video_id: str, folders: Dict[str, Path], 
                     transition_count: int, blank_count: int) -> str:
        """Create README.md explaining folder structure"""
        
        readme_content = f"""# Lecture: {video_id}

## Folder Structure

```
{video_id}/
├── transitions.json          - All deduplicated transition timestamps (including blanks)
├── transition_previews/      - Preview images of all transitions
│   ├── transition_001.jpg    - First transition frame
│   ├── transition_002.jpg    - Second transition frame
│   └── ...
├── slides/                   - (Future) Best quality slide images
│   ├── slide_001.png         - High-res slide 1
│   ├── slide_002.png         - High-res slide 2
│   └── ...
├── audio/                    - (Future) Extracted audio segments
│   ├── slide_001.mp3         - Audio for slide 1
│   ├── slide_002.mp3         - Audio for slide 2
│   └── ...
├── metadata.json            - (Future) Best slide timestamps and quality metrics
└── README.md                - This file
```

## Current Status

- **Total Transitions:** {transition_count}
- **Blank Slides:** {blank_count}
- **Content Slides:** {transition_count - blank_count}

## Files Created

✅ `transitions.json` - Complete transition data with timestamps
✅ `transition_previews/` - All transition frame images (640px width)
⏳ `slides/` - Awaiting high-resolution extraction
⏳ `audio/` - Awaiting FFmpeg audio extraction
⏳ `metadata.json` - Awaiting quality analysis

## Next Steps

1. **Extract Best Slides:**
   - Use `transitions.json` to identify content slides (is_blank=false)
   - Extract high-resolution frames at exact timestamps
   - Apply quality enhancement if needed
   - Save to `slides/` folder

2. **Extract Audio:**
   - Use audio_window start/end from `transitions.json`
   - Run FFmpeg to extract audio segments
   - Save to `audio/` folder

3. **Create Metadata:**
   - Analyze slide quality (sharpness, contrast, text clarity)
   - Add OCR text extraction results
   - Add multimodal AI analysis results
   - Save to `metadata.json`

4. **Multimodal AI Integration:**
   - Feed slide image + audio to GPT-4 Vision / Gemini
   - Generate structured lecture notes
   - Save as `lecture_notes.md` or `notes.json`

## Transition Data Format

```json
{{
  "index": 1,
  "timestamp": 123.45,
  "timestamp_formatted": "02:03",
  "is_blank": false,
  "ssim_score": 0.85,
  "edge_count": 0.75,
  "has_audio": true,
  "audio_window": {{
    "start": 120.0,
    "end": 145.5,
    "duration": 25.5
  }},
  "preview_image": "transition_previews/transition_001.jpg",
  "notes": "Optional manual annotations"
}}
```

## Usage

### Load Transitions
```python
import json

with open('transitions.json', 'r') as f:
    data = json.load(f)

# Get all content transitions (exclude blanks)
content_transitions = [
    t for t in data['transitions'] 
    if not t['is_blank']
]

# Get timestamps for extraction
timestamps = [t['timestamp'] for t in content_transitions]
```

### Extract High-Res Slides
```python
import cv2

cap = cv2.VideoCapture('../videos/{video_id}.mp4')

for trans in content_transitions:
    timestamp = trans['timestamp']
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    
    if ret:
        output_path = f"slides/slide_{{trans['index']:03d}}.png"
        cv2.imwrite(output_path, frame)

cap.release()
```

### Extract Audio with FFmpeg
```bash
# Read audio_window from transitions.json
# For each transition:
ffmpeg -i ../videos/{video_id}.mp4 \\
    -ss <audio_window.start> \\
    -to <audio_window.end> \\
    -vn -acodec libmp3lame -q:a 2 \\
    audio/slide_001.mp3
```

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return readme_content
    
    def organize_video(self, video_id: str, transitions: List[Dict[str, Any]]) -> bool:
        """Organize all content for a single video"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Organizing: {video_id}")
            logger.info(f"{'='*70}")
            
            # Create folder structure
            folders = self.create_folder_structure(video_id)
            logger.info(f"Created folder structure in: {folders['base']}")
            
            # Create transitions.json
            transitions_data = self.create_transitions_json(video_id, transitions)
            transitions_file = folders['base'] / 'transitions.json'
            
            with open(transitions_file, 'w') as f:
                json.dump(transitions_data, f, indent=2)
            
            logger.info(f"Created transitions.json ({len(transitions)} transitions)")
            
            # Find video file
            video_file = None
            for ext in ['.mp4', '.mkv', '.avi', '.mov']:
                potential_file = self.videos_dir / f"{video_id}{ext}"
                if potential_file.exists():
                    video_file = potential_file
                    break
            
            if not video_file:
                logger.warning(f"Video file not found for {video_id}, skipping frame extraction")
            else:
                logger.info(f"Found video: {video_file.name}")
                
                # Extract transition preview frames
                success_count = 0
                for trans in transitions_data['transitions']:
                    output_path = folders['base'] / trans['preview_image']
                    
                    if self.extract_transition_frame(video_file, trans['timestamp'], output_path):
                        success_count += 1
                    
                    if (trans['index']) % 10 == 0:
                        logger.info(f"  Extracted {trans['index']}/{len(transitions)} preview frames...")
                
                logger.info(f"Extracted {success_count}/{len(transitions)} preview frames")
            
            # Create README
            readme_content = self.create_readme(
                video_id, 
                folders, 
                transitions_data['total_transitions'],
                transitions_data['blank_count']
            )
            
            readme_file = folders['base'] / 'README.md'
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            logger.info(f"Created README.md")
            
            # Summary
            logger.info(f"\n✅ Successfully organized {video_id}")
            logger.info(f"   Total transitions: {transitions_data['total_transitions']}")
            logger.info(f"   Blank slides: {transitions_data['blank_count']}")
            logger.info(f"   Content slides: {transitions_data['content_count']}")
            logger.info(f"   Location: {folders['base']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error organizing {video_id}: {e}")
            return False
    
    def organize_all_videos(self) -> Dict[str, Any]:
        """Organize all videos from deduplicated_transitions.json"""
        
        logger.info("\n" + "="*70)
        logger.info("ORGANIZING ALL LECTURE FOLDERS")
        logger.info("="*70)
        
        # Load deduplicated transitions
        with open(self.deduplicated_json, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data['videos'])} videos from deduplicated_transitions.json")
        
        results = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'total_transitions': 0,
            'total_blanks': 0,
            'videos': []
        }
        
        for video_id, transitions in data['videos'].items():
            results['total_videos'] += 1
            results['total_transitions'] += len(transitions)
            results['total_blanks'] += sum(1 for t in transitions if t.get('is_blank', False))
            
            if self.organize_video(video_id, transitions):
                results['successful'] += 1
                results['videos'].append({
                    'video_id': video_id,
                    'status': 'success',
                    'transitions': len(transitions)
                })
            else:
                results['failed'] += 1
                results['videos'].append({
                    'video_id': video_id,
                    'status': 'failed',
                    'transitions': len(transitions)
                })
        
        # Create master index
        self._create_master_index(results)
        
        logger.info("\n" + "="*70)
        logger.info("ORGANIZATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total videos: {results['total_videos']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Total transitions: {results['total_transitions']}")
        logger.info(f"Total blank slides: {results['total_blanks']}")
        logger.info(f"Output location: {self.output_base}")
        
        return results
    
    def _create_master_index(self, results: Dict[str, Any]):
        """Create master index.html for all lectures"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lecture Index - Smart Notes Generator</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .videos-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        .video-card {{
            border: 2px solid #eee;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s;
            background: white;
        }}
        .video-card:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }}
        .video-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
        }}
        .video-stats {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        .video-stat {{
            text-align: center;
        }}
        .video-stat .number {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .video-stat .label {{
            color: #666;
            font-size: 0.85em;
        }}
        .folder-link {{
            display: inline-block;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
            text-align: center;
        }}
        .folder-link:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(17, 153, 142, 0.3);
        }}
        .status-success {{
            color: #11998e;
            font-weight: bold;
        }}
        .status-failed {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Lecture Notes - Smart Notes Generator</h1>
        <div class="subtitle">
            Organized lecture folders with deduplicated transitions, preview images, and preparation for slides & audio extraction
            <br><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="number">{results['total_videos']}</div>
                <div class="label">Total Lectures</div>
            </div>
            <div class="stat-card">
                <div class="number">{results['total_transitions']}</div>
                <div class="label">Total Transitions</div>
            </div>
            <div class="stat-card">
                <div class="number">{results['total_transitions'] - results['total_blanks']}</div>
                <div class="label">Content Slides</div>
            </div>
            <div class="stat-card">
                <div class="number">{results['total_blanks']}</div>
                <div class="label">Blank Slides</div>
            </div>
        </div>
        
        <div class="videos-grid">
"""
        
        for video_info in sorted(results['videos'], key=lambda x: x['video_id']):
            video_id = video_info['video_id']
            status = video_info['status']
            transitions = video_info['transitions']
            
            # Load video-specific data if available
            video_folder = self.output_base / video_id
            transitions_file = video_folder / 'transitions.json'
            
            blank_count = 0
            content_count = transitions
            
            if transitions_file.exists():
                with open(transitions_file, 'r') as f:
                    trans_data = json.load(f)
                    blank_count = trans_data.get('blank_count', 0)
                    content_count = trans_data.get('content_count', transitions)
            
            status_class = 'status-success' if status == 'success' else 'status-failed'
            
            html += f"""
            <div class="video-card">
                <div class="video-title">{video_id}</div>
                <div class="video-stats">
                    <div class="video-stat">
                        <div class="number">{transitions}</div>
                        <div class="label">Transitions</div>
                    </div>
                    <div class="video-stat">
                        <div class="number">{content_count}</div>
                        <div class="label">Content</div>
                    </div>
                    <div class="video-stat">
                        <div class="number">{blank_count}</div>
                        <div class="label">Blanks</div>
                    </div>
                </div>
                <a href="{video_id}/README.md" class="folder-link">📂 Open Lecture Folder</a>
            </div>
"""
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        
        index_file = self.output_base / 'index.html'
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Created master index: {index_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize lecture folders with transitions')
    parser.add_argument('--videos-dir', default='data/videos', help='Videos directory')
    parser.add_argument('--deduplicated-json', 
                       default='data/transition_previews/deduplicated_transitions.json',
                       help='Deduplicated transitions JSON file')
    parser.add_argument('--output-dir', default='data/lectures', help='Output directory for lectures')
    parser.add_argument('--video-id', help='Process single video only')
    
    args = parser.parse_args()
    
    organizer = LectureFolderOrganizer(
        videos_dir=args.videos_dir,
        deduplicated_json=args.deduplicated_json,
        output_base=args.output_dir
    )
    
    if args.video_id:
        # Process single video
        with open(args.deduplicated_json, 'r') as f:
            data = json.load(f)
        
        if args.video_id in data['videos']:
            organizer.organize_video(args.video_id, data['videos'][args.video_id])
        else:
            logger.error(f"Video {args.video_id} not found in deduplicated_transitions.json")
    else:
        # Process all videos
        organizer.organize_all_videos()


if __name__ == '__main__':
    main()
