"""
Compare deduplicated_transitions.json with new deduplication logic output
"""
import json
import pandas as pd
from pathlib import Path
from src.deduplicate_transitions import FinalSlideGenerator
from datetime import datetime

# Load existing deduplicated_transitions.json
existing_file = Path('data/transition_previews/deduplicated_transitions.json')
with open(existing_file, 'r') as f:
    existing_data = json.load(f)

print("="*70)
print("COMPARING DEDUPLICATION LOGIC")
print("="*70)
print(f"\nExisting file: {existing_file}")
print(f"  Generated: {existing_data['generated_at']}")
print(f"  Total videos: {existing_data['total_videos']}")
print(f"  Total slides: {existing_data['total_slides']}")
print(f"  Parameters: {existing_data['parameters']}")

# Get list of videos from all_predictions.csv
predictions_df = pd.read_csv('data/all_predictions.csv')
video_ids = sorted(predictions_df['video_id'].unique())

print(f"\nVideos in all_predictions.csv: {len(video_ids)}")

# Initialize generator with same parameters as existing
params = existing_data['parameters']
generator = FinalSlideGenerator(
    prediction_threshold=params['prediction_threshold'],
    ssim_dedup_threshold=params['ssim_dedup_threshold'],
    ssim_rapid_threshold=params['ssim_rapid_threshold'],
    blank_edge_threshold=params['blank_edge_threshold']
)

# Process all videos with new logic (using predictions.csv)
new_results = {}
total_new_slides = 0

print("\n" + "="*70)
print("PROCESSING WITH NEW LOGIC (using predictions.csv)")
print("="*70)

for idx, video_id in enumerate(video_ids, 1):
    print(f"\n[{idx}/{len(video_ids)}] {video_id}")
    
    # Check if video file exists
    video_path = Path(f'data/videos/{video_id}.mp4')
    if not video_path.exists():
        print(f"  ⚠️ Video not found, skipping")
        continue
    
    # Check if predictions exist in lectures folder
    predictions_file = Path(f'data/lectures/{video_id}/{video_id}_predictions.csv')
    if not predictions_file.exists():
        # Try copying from all_predictions.csv
        video_predictions = predictions_df[predictions_df['video_id'] == video_id]
        if len(video_predictions) > 0:
            lectures_dir = Path(f'data/lectures/{video_id}')
            lectures_dir.mkdir(exist_ok=True, parents=True)
            video_predictions.to_csv(predictions_file, index=False)
            print(f"  ✓ Created predictions file from all_predictions.csv")
        else:
            print(f"  ⚠️ No predictions found, skipping")
            continue
    
    # Check if features exist
    features_file = Path(f'data/output/{video_id}_features.csv')
    if not features_file.exists():
        print(f"  ⚠️ Features not found, skipping")
        continue
    
    try:
        slides = generator.process_video(video_id, use_predictions=True)
        new_results[video_id] = slides
        total_new_slides += len(slides)
        print(f"  ✓ Extracted {len(slides)} slides")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

# Compare results
comparison = []
for video_id in video_ids:
    if video_id not in existing_data['videos']:
        print(f"\n⚠️ {video_id}: Not in existing data")
        continue
    
    if video_id not in new_results:
        print(f"\n⚠️ {video_id}: Not processed with new logic")
        continue
    
    existing_slides = existing_data['videos'][video_id]
    new_slides = new_results[video_id]
    
    existing_count = len(existing_slides)
    new_count = len(new_slides)
    
    # Compare timestamps
    existing_ts = [s['timestamp'] for s in existing_slides]
    new_ts = [s['timestamp'] for s in new_slides]
    
    match = existing_ts == new_ts
    
    comparison.append({
        'video_id': video_id,
        'existing_count': existing_count,
        'new_count': new_count,
        'difference': new_count - existing_count,
        'timestamps_match': match
    })
    
    if match:
        print(f"\n✅ {video_id}: IDENTICAL ({new_count} slides)")
    else:
        print(f"\n❌ {video_id}: DIFFERENT")
        print(f"   Existing: {existing_count} slides")
        print(f"   New:      {new_count} slides")
        print(f"   Diff:     {new_count - existing_count:+d}")
        
        # Show first few differing timestamps
        print(f"\n   First 5 timestamps:")
        print(f"   Existing: {existing_ts[:5]}")
        print(f"   New:      {new_ts[:5]}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

comparison_df = pd.DataFrame(comparison)
print(f"\nTotal videos compared: {len(comparison)}")
print(f"Identical results: {sum(comparison_df['timestamps_match'])}")
print(f"Different results: {sum(~comparison_df['timestamps_match'])}")

print(f"\nExisting total slides: {existing_data['total_slides']}")
print(f"New total slides: {total_new_slides}")
print(f"Difference: {total_new_slides - existing_data['total_slides']:+d}")

if len(comparison_df) > 0:
    print("\nPer-video comparison:")
    print(comparison_df.to_string(index=False))

# Save new results for inspection
new_output = {
    'generated_at': datetime.now().isoformat(),
    'total_videos': len(new_results),
    'total_slides': total_new_slides,
    'parameters': params,
    'videos': new_results
}

output_file = Path('data/transition_previews/deduplicated_transitions_NEW.json')
with open(output_file, 'w') as f:
    json.dump(new_output, f, indent=2)

print(f"\n✓ Saved new results to: {output_file}")
