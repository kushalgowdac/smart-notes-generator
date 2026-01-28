"""
Extract transitions using ONLY clustering logic (burst grouping + best frame selection)
WITHOUT SSIM deduplication - to see pre-deduplication results
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.deduplicate_transitions import FinalSlideGenerator

# Load existing deduplicated_transitions.json for comparison
existing_file = Path('data/transition_previews/deduplicated_transitions.json')
with open(existing_file, 'r') as f:
    existing_data = json.load(f)

print("="*70)
print("EXTRACTING WITH CLUSTERING LOGIC ONLY")
print("="*70)
print("\nThis will use:")
print("  1. Prediction threshold filtering (0.1)")
print("  2. Burst grouping (consecutive predictions)")
print("  3. Best frame selection (lowest SSIM within burst)")
print("  4. NO SSIM deduplication between slides")
print()

# Get parameters
params = existing_data['parameters']

# Get list of videos from all_predictions.csv
predictions_df = pd.read_csv('data/all_predictions.csv')
video_ids = sorted(predictions_df['video_id'].unique())

print(f"Videos to process: {len(video_ids)}")

# Process all videos with clustering only
clustering_results = {}
total_slides = 0

for idx, video_id in enumerate(video_ids, 1):
    print(f"\n[{idx}/{len(video_ids)}] {video_id}")
    
    # Check if predictions exist
    video_predictions = predictions_df[predictions_df['video_id'] == video_id]
    if len(video_predictions) == 0:
        print(f"  ⚠️ No predictions found, skipping")
        continue
    
    # Filter by prediction threshold
    positive_predictions = video_predictions[
        video_predictions['smoothed_prediction'] >= params['prediction_threshold']
    ].copy()
    
    print(f"  Predictions: {len(video_predictions)} total, {len(positive_predictions)} positive")
    
    # Group consecutive predictions into bursts
    timestamps = sorted(positive_predictions['timestamp_seconds'].unique())
    
    if len(timestamps) == 0:
        print(f"  ⚠️ No positive predictions, skipping")
        continue
    
    bursts = []
    current_burst = {
        'start_time': timestamps[0],
        'end_time': timestamps[0],
        'timestamps': [timestamps[0]]
    }
    
    for i in range(1, len(timestamps)):
        time_gap = timestamps[i] - timestamps[i-1]
        
        if time_gap <= 0.5:  # Within 0.5s = same burst
            current_burst['end_time'] = timestamps[i]
            current_burst['timestamps'].append(timestamps[i])
        else:
            # Save current burst and start new one
            bursts.append(current_burst)
            current_burst = {
                'start_time': timestamps[i],
                'end_time': timestamps[i],
                'timestamps': [timestamps[i]]
            }
    
    # Add final burst
    bursts.append(current_burst)
    
    print(f"  Bursts: {len(bursts)} groups")
    
    # Select best frame from each burst (lowest SSIM = highest change)
    # Check if features exist
    features_file = Path(f'data/output/{video_id}_features.csv')
    if not features_file.exists():
        print(f"  ⚠️ Features not found, using middle timestamp fallback")
        candidate_timestamps = [burst['timestamps'][len(burst['timestamps']) // 2] for burst in bursts]
    else:
        features_df = pd.read_csv(features_file)
        candidate_timestamps = []
        
        for burst in bursts:
            # Get features for all timestamps in burst
            burst_features = features_df[
                features_df['timestamp_seconds'].isin(burst['timestamps'])
            ].copy()
            
            if len(burst_features) == 0:
                # Fallback to middle of burst
                best_ts = burst['timestamps'][len(burst['timestamps']) // 2]
            else:
                # Find frame with lowest SSIM (highest change)
                best_row = burst_features.loc[burst_features['global_ssim'].idxmin()]
                best_ts = best_row['timestamp_seconds']
            
            candidate_timestamps.append(best_ts)
    
    print(f"  Selected: {len(candidate_timestamps)} candidate transitions")
    
    # Create slides WITHOUT SSIM deduplication
    slides = []
    for i, ts in enumerate(candidate_timestamps, 1):
        slide = {
            'slide_number': i,
            'timestamp': round(float(ts), 2),
            'audio_window_start': round(float(candidate_timestamps[i-2]) if i > 1 else 0.0, 2),
            'audio_window_end': round(float(ts), 2)
        }
        slide['audio_duration'] = round(slide['audio_window_end'] - slide['audio_window_start'], 2)
        slides.append(slide)
    
    clustering_results[video_id] = slides
    total_slides += len(slides)
    print(f"  ✓ Result: {len(slides)} slides (NO deduplication applied)")

print("\n" + "="*70)
print("COMPARISON WITH EXISTING")
print("="*70)

# Compare results
comparison = []
for video_id in video_ids:
    if video_id not in existing_data['videos']:
        continue
    
    if video_id not in clustering_results:
        continue
    
    existing_slides = existing_data['videos'][video_id]
    clustering_slides = clustering_results[video_id]
    
    existing_count = len(existing_slides)
    clustering_count = len(clustering_slides)
    
    comparison.append({
        'video_id': video_id,
        'existing_count': existing_count,
        'clustering_count': clustering_count,
        'difference': clustering_count - existing_count
    })
    
    diff = clustering_count - existing_count
    if diff == 0:
        print(f"\n✅ {video_id}: SAME ({clustering_count} slides)")
    else:
        print(f"\n{'⬆️' if diff > 0 else '⬇️'} {video_id}: {existing_count} → {clustering_count} ({diff:+d})")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

comparison_df = pd.DataFrame(comparison)
print(f"\nTotal videos: {len(comparison)}")
print(f"\nExisting (with SSIM dedup @ 0.92): {existing_data['total_slides']} slides")
print(f"Clustering only (no SSIM dedup): {total_slides} slides")
print(f"Difference: {total_slides - existing_data['total_slides']:+d} slides")

if len(comparison_df) > 0:
    print("\nPer-video breakdown:")
    print(comparison_df.to_string(index=False))

# Save clustering-only results
output = {
    'generated_at': datetime.now().isoformat(),
    'method': 'clustering_only',
    'description': 'Burst grouping + best frame selection WITHOUT SSIM deduplication',
    'total_videos': len(clustering_results),
    'total_slides': total_slides,
    'parameters': {
        'prediction_threshold': params['prediction_threshold'],
        'burst_window': 0.5,
        'selection_method': 'lowest_ssim_in_burst'
    },
    'videos': clustering_results
}

output_file = Path('data/transition_previews/deduplicated_transitions_CLUSTERING_ONLY.json')
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved clustering-only results to: {output_file}")
