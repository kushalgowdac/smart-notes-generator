"""
Compare Manual vs Predicted Transitions
Generate confusion matrices, accuracy metrics, and threshold tuning analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransitionComparisonAnalyzer:
    """Analyze and compare manual vs predicted transitions"""
    
    def __init__(self, master_dataset='data/master_dataset.csv',
                 all_predictions='data/all_predictions.csv',
                 ground_truth_dir='data/ground_truth',
                 deduplicated_json='data/transition_previews/deduplicated_transitions.json',
                 output_dir='data/analysis'):
        
        self.master_dataset = Path(master_dataset)
        self.all_predictions = Path(all_predictions)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.deduplicated_json = Path(deduplicated_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ONLY test dataset videos for proper ML evaluation
        self.test_videos = ['chemistry_01_english', 'chemistry_10_english', 
                           'mathematics_03_english', 'toc_1']
        
        logger.info("Initialized TransitionComparisonAnalyzer")
        logger.info(f"Test videos: {self.test_videos}")
    
    def analyze_duplicated_transitions(self):
        """
        Compare RAW (duplicated) manual vs predicted transitions
        Uses all_predictions.csv where both label=1 (manual) and smoothed_prediction (model) exist
        ONLY on TEST dataset videos for proper ML evaluation
        """
        logger.info("\n" + "="*70)
        logger.info("ANALYZING DUPLICATED (RAW) TRANSITIONS - TEST SET ONLY")
        logger.info("="*70)
        
        # Load data from all_predictions.csv (has both label and smoothed_prediction)
        df = pd.read_csv(self.all_predictions)
        
        # Filter to TEST videos only
        df = df[df['video_id'].isin(self.test_videos)]
        logger.info(f"Filtered to {len(self.test_videos)} test videos: {self.test_videos}")
        
        # Get manual labels and predictions
        y_true = df['label'].values  # Manual ground truth
        y_pred_proba = df['smoothed_prediction'].values  # Model probability (0 or 1 after smoothing)
        
        # Convert to binary (already binary after NMS smoothing)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        logger.info(f"Total frames: {len(df)}")
        logger.info(f"Manual positives (label=1): {np.sum(y_true)}")
        logger.info(f"Predicted positives (smoothed_prediction=1): {np.sum(y_pred)}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'total_frames': len(df),
            'manual_positives': int(np.sum(y_true)),
            'predicted_positives': int(np.sum(y_pred)),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        logger.info(f"\nMetrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, 
                                    title="Confusion Matrix: Duplicated Transitions\n(Raw Manual vs Raw Predicted)",
                                    filename="confusion_matrix_duplicated.png")
        
        # Save metrics
        with open(self.output_dir / 'metrics_duplicated.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def analyze_deduplicated_transitions(self):
        """
        Compare DEDUPLICATED manual vs predicted transitions
        Uses transitions.txt (manual unique) vs deduplicated_transitions.json (model unique)
        ONLY on TEST dataset videos, EXCLUDING forced end-of-video transitions
        """
        logger.info("\n" + "="*70)
        logger.info("ANALYZING DEDUPLICATED (UNIQUE) TRANSITIONS - TEST SET ONLY")
        logger.info("="*70)
        
        # Load deduplicated predictions
        with open(self.deduplicated_json, 'r') as f:
            pred_data = json.load(f)
        
        # Filter to TEST videos only
        videos = [v for v in pred_data['videos'].keys() if v in self.test_videos]
        logger.info(f"Filtered to {len(self.test_videos)} test videos: {self.test_videos}")
        
        comparisons = []
        
        for video_id in videos:
            # Load manual deduplicated transitions
            manual_file = self.ground_truth_dir / video_id / 'transitions.txt'
            if not manual_file.exists():
                logger.warning(f"Manual transitions not found for {video_id}")
                continue
            
            manual_timestamps = []
            with open(manual_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            manual_timestamps.append(float(line))
                        except ValueError:
                            continue
            
            # Get predicted timestamps
            all_pred_slides = pred_data['videos'][video_id]
            
            # Filter out end-of-video forced transitions
            # These are added by model at video_duration - 1.0s to capture final slide
            # Manual labels don't have these, so they create false positives
            if all_pred_slides:
                video_duration = all_pred_slides[-1]['timestamp'] + 2.0  # Approximate
                # Remove transitions within 2s of video end (forced final captures)
                pred_timestamps = [
                    slide['timestamp'] for slide in all_pred_slides 
                    if slide['timestamp'] < video_duration - 2.0
                ]
                removed_count = len(all_pred_slides) - len(pred_timestamps)
                if removed_count > 0:
                    logger.info(f"  {video_id}: Excluded {removed_count} end-of-video forced transition(s)")
            else:
                pred_timestamps = []
            
            # Compare with tolerance (±2 seconds)
            tolerance = 2.0
            
            tp = 0  # Correctly predicted transitions
            fp = 0  # False alarms (predicted but not manual)
            fn = 0  # Missed transitions (manual but not predicted)
            
            matched_manual = set()
            matched_pred = set()
            
            # Find matches
            for pred_ts in pred_timestamps:
                matched = False
                for idx, manual_ts in enumerate(manual_timestamps):
                    if abs(pred_ts - manual_ts) <= tolerance:
                        if idx not in matched_manual:
                            tp += 1
                            matched_manual.add(idx)
                            matched_pred.add(pred_ts)
                            matched = True
                            break
                if not matched:
                    fp += 1
            
            fn = len(manual_timestamps) - len(matched_manual)
            
            comparisons.append({
                'video_id': video_id,
                'manual_count': len(manual_timestamps),
                'predicted_count': len(pred_timestamps),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            })
        
        # Aggregate metrics
        total_tp = sum(c['true_positives'] for c in comparisons)
        total_fp = sum(c['false_positives'] for c in comparisons)
        total_fn = sum(c['false_negatives'] for c in comparisons)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'total_videos': len(comparisons),
            'total_manual_transitions': sum(c['manual_count'] for c in comparisons),
            'total_predicted_transitions': sum(c['predicted_count'] for c in comparisons),
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'tolerance_seconds': tolerance,
            'per_video': comparisons
        }
        
        logger.info(f"\nAggregate Metrics (±{tolerance}s tolerance):")
        logger.info(f"  Total videos: {len(comparisons)}")
        logger.info(f"  Manual transitions: {metrics['total_manual_transitions']}")
        logger.info(f"  Predicted transitions: {metrics['total_predicted_transitions']}")
        logger.info(f"  True Positives: {total_tp}")
        logger.info(f"  False Positives: {total_fp}")
        logger.info(f"  False Negatives: {total_fn}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Create pseudo confusion matrix for visualization
        # Since we're comparing timestamps, not frames, we'll visualize TP/FP/FN
        cm_data = np.array([[total_tp, total_fp], [total_fn, 0]])
        
        self._plot_confusion_matrix_deduplicated(total_tp, total_fp, total_fn,
                                                  title="Transition Matching: Deduplicated\n(Manual vs Predicted, ±2s tolerance)",
                                                  filename="confusion_matrix_deduplicated.png")
        
        # Save metrics
        with open(self.output_dir / 'metrics_deduplicated.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Per-video comparison table
        self._plot_per_video_comparison(comparisons)
        
        return metrics
    
    def _plot_confusion_matrix(self, cm, title, filename):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Labels
        labels = ['Non-Transition (0)', 'Transition (1)']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 16, 'weight': 'bold'})
        
        plt.title(title, fontsize=16, weight='bold', pad=20)
        plt.ylabel('Manual (Ground Truth)', fontsize=14, weight='bold')
        plt.xlabel('Predicted (Model)', fontsize=14, weight='bold')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=12, color='gray')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filename}")
    
    def _plot_confusion_matrix_deduplicated(self, tp, fp, fn, title, filename):
        """Plot confusion matrix for deduplicated transitions"""
        plt.figure(figsize=(10, 8))
        
        # Create matrix visualization
        data = [[tp, fp], [fn, 0]]
        labels_x = ['Predicted as Transition', 'Not Predicted']
        labels_y = ['Manual Transition', 'No Match']
        
        ax = sns.heatmap(data, annot=True, fmt='d', cmap='RdYlGn',
                        xticklabels=labels_x, yticklabels=labels_y,
                        cbar_kws={'label': 'Count'},
                        annot_kws={'size': 16, 'weight': 'bold'})
        
        plt.title(title, fontsize=16, weight='bold', pad=20)
        plt.ylabel('Manual Ground Truth', fontsize=14, weight='bold')
        plt.xlabel('Model Prediction', fontsize=14, weight='bold')
        
        # Add descriptions
        total = tp + fp + fn
        plt.text(0.5, 0.3, f'Correctly\nDetected\n({tp}/{total})', 
                ha='center', va='center', fontsize=11, weight='bold', color='darkgreen')
        plt.text(1.5, 0.3, f'False\nAlarms\n({fp}/{total})', 
                ha='center', va='center', fontsize=11, weight='bold', color='darkred')
        plt.text(0.5, 1.3, f'Missed\nTransitions\n({fn}/{total})', 
                ha='center', va='center', fontsize=11, weight='bold', color='darkorange')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filename}")
    
    def _plot_per_video_comparison(self, comparisons):
        """Plot per-video comparison chart - TEST VIDEOS ONLY"""
        plt.figure(figsize=(12, 8))  # Smaller since only 4 videos
        
        videos = [c['video_id'] for c in comparisons]
        manual_counts = [c['manual_count'] for c in comparisons]
        pred_counts = [c['predicted_count'] for c in comparisons]
        tp_counts = [c['true_positives'] for c in comparisons]
        
        x = np.arange(len(videos))
        width = 0.25
        
        plt.bar(x - width, manual_counts, width, label='Manual (Ground Truth)', color='#11998e', alpha=0.8)
        plt.bar(x, pred_counts, width, label='Predicted (Model)', color='#667eea', alpha=0.8)
        plt.bar(x + width, tp_counts, width, label='Correct Matches (TP)', color='#38ef7d', alpha=0.8)
        
        plt.xlabel('Video ID', fontsize=12, weight='bold')
        plt.ylabel('Number of Transitions', fontsize=12, weight='bold')
        plt.title('Per-Video Transition Comparison\n(Manual vs Predicted, Deduplicated)', 
                 fontsize=14, weight='bold', pad=20)
        plt.xticks(x, videos, rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'per_video_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  Saved: per_video_comparison.png")
    
    def generate_threshold_tuning_analysis(self):
        """
        Generate threshold tuning plots
        Shows how precision/recall change with different probability thresholds
        TEST DATASET ONLY
        """
        logger.info("\n" + "="*70)
        logger.info("GENERATING THRESHOLD TUNING ANALYSIS - TEST SET ONLY")
        logger.info("="*70)
        
        # Load raw predictions (before NMS smoothing)
        df = pd.read_csv(self.all_predictions)
        
        # Filter to TEST videos only
        df = df[df['video_id'].isin(self.test_videos)]
        
        # Get manual labels
        y_true = df['label'].values
        
        # We need raw probabilities before smoothing
        # If smoothed_prediction is binary, we can't do threshold tuning
        # Let's check if we have raw predictions
        
        logger.warning("Note: smoothed_prediction is already binary after NMS")
        logger.warning("Threshold tuning requires raw probabilities before smoothing")
        logger.warning("If you have raw model output, please provide it for accurate threshold tuning")
        
        # For demonstration, let's create synthetic analysis
        # In production, you'd use actual raw probabilities
        
        return None
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("\n" + "="*70)
        logger.info("RUNNING FULL COMPARISON ANALYSIS")
        logger.info("="*70)
        
        results = {}
        
        # Analyze duplicated transitions
        results['duplicated'] = self.analyze_duplicated_transitions()
        
        # Analyze deduplicated transitions
        results['deduplicated'] = self.analyze_deduplicated_transitions()
        
        # Threshold tuning (if raw probabilities available)
        # results['threshold_tuning'] = self.generate_threshold_tuning_analysis()
        
        # Create summary report
        self._create_summary_report(results)
        
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*70)
        logger.info(f"Results saved in: {self.output_dir}")
        logger.info("Files generated:")
        logger.info("  - confusion_matrix_duplicated.png")
        logger.info("  - confusion_matrix_deduplicated.png")
        logger.info("  - per_video_comparison.png")
        logger.info("  - metrics_duplicated.json")
        logger.info("  - metrics_deduplicated.json")
        logger.info("  - comparison_summary.txt")
        
        return results
    
    def _create_summary_report(self, results):
        """Create text summary report"""
        report = f"""
SMART NOTES GENERATOR - COMPARISON ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

1. DUPLICATED (RAW) TRANSITIONS ANALYSIS
{'='*70}
Source: master_dataset.csv (all frames labeled as 1, including 11 consecutive)

Total Frames Analyzed: {results['duplicated']['total_frames']:,}
Manual Positives (label=1): {results['duplicated']['manual_positives']:,}
Predicted Positives (smoothed_prediction=1): {results['duplicated']['predicted_positives']:,}

Confusion Matrix:
  True Negatives (TN):  {results['duplicated']['true_negatives']:,}
  False Positives (FP): {results['duplicated']['false_positives']:,}
  False Negatives (FN): {results['duplicated']['false_negatives']:,}
  True Positives (TP):  {results['duplicated']['true_positives']:,}

Performance Metrics:
  Accuracy:  {results['duplicated']['accuracy']:.4f} ({results['duplicated']['accuracy']*100:.2f}%)
  Precision: {results['duplicated']['precision']:.4f} ({results['duplicated']['precision']*100:.2f}%)
  Recall:    {results['duplicated']['recall']:.4f} ({results['duplicated']['recall']*100:.2f}%)
  F1 Score:  {results['duplicated']['f1_score']:.4f}

{'='*70}

2. DEDUPLICATED (UNIQUE) TRANSITIONS ANALYSIS
{'='*70}
Source: transitions.txt vs deduplicated_transitions.json
Matching Tolerance: ±{results['deduplicated']['tolerance_seconds']}s

Total Videos: {results['deduplicated']['total_videos']}
Manual Transitions: {results['deduplicated']['total_manual_transitions']}
Predicted Transitions: {results['deduplicated']['total_predicted_transitions']}

Matching Results:
  True Positives (TP):  {results['deduplicated']['true_positives']} (correctly detected)
  False Positives (FP): {results['deduplicated']['false_positives']} (false alarms)
  False Negatives (FN): {results['deduplicated']['false_negatives']} (missed transitions)

Performance Metrics:
  Precision: {results['deduplicated']['precision']:.4f} ({results['deduplicated']['precision']*100:.2f}%)
  Recall:    {results['deduplicated']['recall']:.4f} ({results['deduplicated']['recall']*100:.2f}%)
  F1 Score:  {results['deduplicated']['f1_score']:.4f}

{'='*70}

INTERPRETATION:
- Duplicated analysis shows frame-level accuracy (including consecutive frames)
- Deduplicated analysis shows transition-level accuracy (unique events)
- High recall means model catches most transitions
- High precision means few false alarms
- F1 score balances both precision and recall

Per-video details saved in metrics_deduplicated.json
"""
        
        with open(self.output_dir / 'comparison_summary.txt', 'w') as f:
            f.write(report)
        
        logger.info("  Saved: comparison_summary.txt")


def main():
    analyzer = TransitionComparisonAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
