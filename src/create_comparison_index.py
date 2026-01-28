"""
Create comprehensive comparison index HTML page
Shows all videos with links to manual vs predicted transitions
"""

import json
from pathlib import Path
from datetime import datetime

def create_comparison_index():
    """Create master comparison index page"""
    
    output_dir = Path('data/transition_previews')
    
    # Load deduplicated transitions data
    with open(output_dir / 'deduplicated_transitions.json', 'r') as f:
        dedup_data = json.load(f)
    
    # Get all videos
    videos = sorted(dedup_data['videos'].keys())
    
    # Load analysis metrics
    analysis_dir = Path('data/analysis')
    with open(analysis_dir / 'metrics_deduplicated.json', 'r') as f:
        metrics = json.load(f)
    
    # Create HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Notes Generator - Comparison Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        h1 {
            color: #333;
            font-size: 36px;
            margin-bottom: 15px;
        }
        .subtitle {
            color: #666;
            font-size: 18px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 25px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
        }
        .section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            margin-bottom: 30px;
        }
        h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }
        .video-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        .video-card:hover {
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        .video-name {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
        }
        .links-grid {
            display: grid;
            gap: 10px;
        }
        .link-group {
            margin-bottom: 10px;
        }
        .link-group-title {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .link-btn {
            display: inline-block;
            padding: 10px 15px;
            margin: 3px;
            border-radius: 8px;
            text-decoration: none;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .btn-manual {
            background: #11998e;
            color: white;
        }
        .btn-manual:hover {
            background: #0f8a7f;
            transform: translateY(-2px);
        }
        .btn-predicted {
            background: #667eea;
            color: white;
        }
        .btn-predicted:hover {
            background: #5568d3;
            transform: translateY(-2px);
        }
        .btn-raw {
            background: #f39c12;
            color: white;
        }
        .btn-raw:hover {
            background: #d68910;
            transform: translateY(-2px);
        }
        .stats {
            font-size: 13px;
            color: #666;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
        }
        .analysis-links {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .analysis-btn {
            padding: 15px 25px;
            background: #764ba2;
            color: white;
            text-decoration: none;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .analysis-btn:hover {
            background: #6a4290;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Smart Notes Generator - Comparison Dashboard</h1>
            <p class="subtitle">Compare Manual (Ground Truth) vs Predicted Transitions</p>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Videos</div>
                    <div class="metric-value">""" + str(len(videos)) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Manual Transitions</div>
                    <div class="metric-value">""" + str(metrics['total_manual_transitions']) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Predicted Transitions</div>
                    <div class="metric-value">""" + str(metrics['total_predicted_transitions']) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value">""" + f"{metrics['recall']*100:.1f}%" + """</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Analysis Results</h2>
            <div class="analysis-links">
                <a href="../analysis/confusion_matrix_duplicated.png" class="analysis-btn" target="_blank">
                    📈 Confusion Matrix (Duplicated)
                </a>
                <a href="../analysis/confusion_matrix_deduplicated.png" class="analysis-btn" target="_blank">
                    📊 Confusion Matrix (Deduplicated)
                </a>
                <a href="../analysis/per_video_comparison.png" class="analysis-btn" target="_blank">
                    📉 Per-Video Comparison Chart
                </a>
                <a href="../analysis/comparison_summary.txt" class="analysis-btn" target="_blank">
                    📄 Full Report (TXT)
                </a>
            </div>
        </div>
        
        <div class="section">
            <h2>🎬 Video Comparisons (""" + str(len(videos)) + """ videos)</h2>
            <div class="video-grid">
"""
    
    # Add each video
    for video_id in videos:
        video_metrics = next((v for v in metrics['per_video'] if v['video_id'] == video_id), None)
        
        manual_count = video_metrics['manual_count'] if video_metrics else 0
        pred_count = video_metrics['predicted_count'] if video_metrics else 0
        tp = video_metrics['true_positives'] if video_metrics else 0
        
        html += f"""
                <div class="video-card">
                    <div class="video-name">{video_id}</div>
                    
                    <div class="link-group">
                        <div class="link-group-title">🎯 DEDUPLICATED (Unique Transitions)</div>
                        <a href="{video_id}/{video_id}_manual_deduplicated_transitions.html" class="link-btn btn-manual" target="_blank">
                            Manual ({manual_count})
                        </a>
                        <a href="{video_id}/{video_id}_deduplicated_transitions.html" class="link-btn btn-predicted" target="_blank">
                            Predicted ({pred_count})
                        </a>
                    </div>
                    
                    <div class="link-group">
                        <div class="link-group-title">📦 RAW (All Detections)</div>
                        <a href="{video_id}/{video_id}_manual_raw_transitions.html" class="link-btn btn-raw" target="_blank">
                            Manual Raw
                        </a>
                        <a href="{video_id}/{video_id}_transitions.html" class="link-btn btn-raw" target="_blank">
                            Predicted Raw
                        </a>
                    </div>
                    
                    <div class="stats">
                        ✅ Matches: {tp} | ❌ False Alarms: {video_metrics['false_positives'] if video_metrics else 0} | ⚠️ Missed: {video_metrics['false_negatives'] if video_metrics else 0}
                    </div>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>ℹ️ Legend</h2>
            <p style="line-height: 1.8; color: #666;">
                <strong style="color: #11998e;">Manual (Ground Truth):</strong> Human-labeled transitions from transitions.txt files<br>
                <strong style="color: #667eea;">Predicted (Model):</strong> XGBoost model predictions with NMS smoothing<br>
                <strong style="color: #f39c12;">Raw:</strong> All detections including consecutive frames (11-frame labeling)<br>
                <strong>Deduplicated:</strong> Unique transitions after SSIM deduplication (±2s tolerance for matching)
            </p>
        </div>
        
        <footer style="text-align: center; color: white; padding: 20px;">
            <strong>Smart Notes Generator v1.5.0</strong><br>
            Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
        </footer>
    </div>
</body>
</html>
"""
    
    # Save
    output_path = output_dir / 'comparison_index.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Comparison index created: {output_path}")
    print(f"   Total videos: {len(videos)}")
    print(f"   Manual transitions: {metrics['total_manual_transitions']}")
    print(f"   Predicted transitions: {metrics['total_predicted_transitions']}")
    print(f"   Accuracy: {metrics['recall']*100:.2f}%")

if __name__ == "__main__":
    create_comparison_index()
