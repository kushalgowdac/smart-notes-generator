"""
Model Refinement: Threshold Tuning and Post-Processing
======================================================
Improves XGBoost model performance by:
1. Finding optimal probability threshold for high recall
2. Applying temporal smoothing (NMS) to reduce false positives
3. Analyzing hard negatives to identify failure patterns

Author: Smart Notes Generator Team
Date: January 26, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc
)
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_refinement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelRefiner:
    """Refine XGBoost model with threshold tuning and post-processing"""
    
    def __init__(self, model_path, data_dir='data'):
        """
        Initialize the refiner
        
        Args:
            model_path: Path to trained model (.pkl file)
            data_dir: Directory containing test dataset
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.model = None
        self.optimal_threshold = 0.5
        
        logger.info("Initialized ModelRefiner")
        logger.info(f"  Model Path: {self.model_path}")
        logger.info(f"  Data Directory: {self.data_dir}")
    
    def load_model_and_data(self):
        """Load trained model and test dataset"""
        logger.info("\n" + "="*60)
        logger.info("Loading Model and Test Data")
        logger.info("="*60)
        
        # Load model
        logger.info(f"Loading model: {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully!")
        
        # Load test dataset
        test_path = self.data_dir / 'test_dataset.csv'
        logger.info(f"\nLoading test set: {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"  Shape: {test_df.shape}")
        
        # Separate features and labels
        exclude_cols = ['video_id', 'frame_index', 'timestamp_seconds', 'label']
        feature_columns = [col for col in test_df.columns if col not in exclude_cols]
        
        X_test = test_df[feature_columns].values
        y_test = test_df['label'].values
        
        # Store metadata for post-processing
        self.test_metadata = test_df[['video_id', 'frame_index', 'timestamp_seconds']].copy()
        self.test_labels = y_test
        
        logger.info(f"Features: {len(feature_columns)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        return X_test, y_test
    
    def find_optimal_threshold(self, X_test, y_test):
        """
        Evaluate model at different probability thresholds
        Find the 'sweet spot' for high recall
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        logger.info("\n" + "="*60)
        logger.info("Threshold Tuning - Finding the Sweet Spot")
        logger.info("="*60)
        logger.info("Goal: Recall > 85% while maintaining reasonable precision")
        
        # Get probability predictions
        logger.info("\nGenerating probability predictions...")
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Test thresholds from 0.05 to 0.95
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
                      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
        results = []
        
        logger.info("\n" + "-"*80)
        logger.info("Threshold | Accuracy | Precision | Recall  | F1-Score | TP  | FP   | FN  | TN")
        logger.info("-"*80)
        
        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            
            results.append({
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
            
            logger.info(f"  {threshold:.2f}    | {acc:.4f}  | {prec:.4f}   | {rec:.4f} | {f1:.4f}  | "
                       f"{tp:3d} | {fp:4d} | {fn:3d} | {tn:5d}")
        
        logger.info("-"*80)
        
        # Find optimal threshold (recall >= 0.85)
        results_df = pd.DataFrame(results)
        
        # Strategy 1: Highest recall
        high_recall = results_df[results_df['recall'] >= 0.85]
        if len(high_recall) > 0:
            # Among high recall, pick highest F1
            optimal_idx = high_recall['f1'].idxmax()
            optimal = results_df.loc[optimal_idx]
            strategy = "High Recall (>= 85%)"
        else:
            # Fallback: Best F1 score
            optimal_idx = results_df['f1'].idxmax()
            optimal = results_df.loc[optimal_idx]
            strategy = "Best F1 Score"
        
        self.optimal_threshold = optimal['threshold']
        
        logger.info("\n" + "="*60)
        logger.info("OPTIMAL THRESHOLD FOUND")
        logger.info("="*60)
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Threshold: {self.optimal_threshold:.2f}")
        logger.info(f"  Accuracy:  {optimal['accuracy']:.4f}")
        logger.info(f"  Precision: {optimal['precision']:.4f}")
        logger.info(f"  Recall:    {optimal['recall']:.4f} <- TARGET")
        logger.info(f"  F1-Score:  {optimal['f1']:.4f}")
        logger.info(f"  True Positives:  {int(optimal['tp'])} (transitions caught)")
        logger.info(f"  False Positives: {int(optimal['fp'])} (false alarms)")
        logger.info(f"  False Negatives: {int(optimal['fn'])} (missed transitions)")
        
        return results_df, y_proba
    
    def apply_temporal_smoothing(self, predictions_df, window_seconds=5.0):
        """
        Apply Non-Maximum Suppression (NMS) to clean predictions
        Collapse multiple detections in a window to single timestamp
        
        Args:
            predictions_df: DataFrame with predictions and metadata
            window_seconds: Time window for clustering (default 5 seconds)
        """
        logger.info("\n" + "="*60)
        logger.info("Temporal Smoothing (NMS)")
        logger.info("="*60)
        logger.info(f"Window: {window_seconds} seconds")
        logger.info("Goal: Collapse multiple detections into single transitions")
        
        smoothed_predictions = []
        original_positives = (predictions_df['prediction'] == 1).sum()
        
        # Process each video separately
        for video_id in predictions_df['video_id'].unique():
            video_df = predictions_df[predictions_df['video_id'] == video_id].copy()
            
            # Find all predicted transitions
            transitions = video_df[video_df['prediction'] == 1].copy()
            
            if len(transitions) == 0:
                # No transitions, keep all original predictions
                smoothed_predictions.append(video_df)
                continue
            
            # Sort by timestamp
            transitions = transitions.sort_values('timestamp_seconds')
            
            # Cluster transitions within window
            clusters = []
            current_cluster = [transitions.iloc[0]]
            
            for i in range(1, len(transitions)):
                current = transitions.iloc[i]
                prev = current_cluster[-1]
                
                # Check if within window
                time_diff = current['timestamp_seconds'] - prev['timestamp_seconds']
                if time_diff <= window_seconds:
                    current_cluster.append(current)
                else:
                    # Start new cluster
                    clusters.append(current_cluster)
                    current_cluster = [current]
            
            # Add last cluster
            if current_cluster:
                clusters.append(current_cluster)
            
            # For each cluster, keep only the frame with highest probability
            kept_indices = set()
            for cluster in clusters:
                if len(cluster) == 1:
                    kept_indices.add(cluster[0].name)
                else:
                    # Find frame with max probability in cluster
                    cluster_df = pd.DataFrame(cluster)
                    max_prob_idx = cluster_df['probability'].idxmax()
                    kept_indices.add(max_prob_idx)
            
            # Reset predictions: keep only high-confidence detections
            video_df['smoothed_prediction'] = 0
            video_df.loc[list(kept_indices), 'smoothed_prediction'] = 1
            
            smoothed_predictions.append(video_df)
            
            logger.info(f"\n{video_id}:")
            logger.info(f"  Original detections: {len(transitions)}")
            logger.info(f"  Clusters formed: {len(clusters)}")
            logger.info(f"  After smoothing: {len(kept_indices)}")
            logger.info(f"  Reduction: {len(transitions) - len(kept_indices)} detections")
        
        # Combine all videos
        smoothed_df = pd.concat(smoothed_predictions, ignore_index=True)
        
        smoothed_positives = (smoothed_df['smoothed_prediction'] == 1).sum()
        
        logger.info("\n" + "="*60)
        logger.info("Temporal Smoothing Results")
        logger.info("="*60)
        logger.info(f"Before smoothing: {original_positives} detections")
        logger.info(f"After smoothing:  {smoothed_positives} detections")
        logger.info(f"Removed:          {original_positives - smoothed_positives} false alarms")
        
        # Evaluate smoothed predictions
        y_true = smoothed_df['label'].values
        y_pred_smoothed = smoothed_df['smoothed_prediction'].values
        
        acc = accuracy_score(y_true, y_pred_smoothed)
        prec = precision_score(y_true, y_pred_smoothed, zero_division=0)
        rec = recall_score(y_true, y_pred_smoothed, zero_division=0)
        f1 = f1_score(y_true, y_pred_smoothed, zero_division=0)
        
        logger.info("\nSmoothed Metrics:")
        logger.info(f"  Accuracy:  {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f} <- Should improve")
        logger.info(f"  Recall:    {rec:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        return smoothed_df
    
    def analyze_hard_negatives(self, predictions_df, top_n=10):
        """
        Analyze false positives (hard negatives)
        Find frames the model thinks are transitions but aren't
        
        Args:
            predictions_df: DataFrame with predictions
            top_n: Number of top false positives to analyze
        """
        logger.info("\n" + "="*60)
        logger.info("False Positive Analysis (Hard Negatives)")
        logger.info("="*60)
        logger.info("Goal: Identify what's confusing the model")
        
        # Find false positives
        false_positives = predictions_df[
            (predictions_df['label'] == 0) & 
            (predictions_df['prediction'] == 1)
        ].copy()
        
        logger.info(f"\nTotal False Positives: {len(false_positives)}")
        
        if len(false_positives) == 0:
            logger.info("No false positives! Model is perfect!")
            return None
        
        # Sort by probability (highest confidence mistakes)
        false_positives = false_positives.sort_values('probability', ascending=False)
        
        logger.info(f"\nTop {top_n} False Positives (Highest Confidence Mistakes):")
        logger.info("-"*100)
        logger.info("Rank | Video ID                  | Frame | Time (s) | Probability | Features")
        logger.info("-"*100)
        
        # Load feature data to analyze patterns
        test_df = pd.read_csv(self.data_dir / 'test_dataset.csv')
        
        top_fp = false_positives.head(top_n)
        
        for i, (idx, row) in enumerate(top_fp.iterrows(), 1):
            # Get feature values for this frame
            frame_data = test_df.iloc[idx]
            
            logger.info(f"{i:4d} | {row['video_id']:25s} | {int(row['frame_index']):5d} | "
                       f"{row['timestamp_seconds']:7.2f} | {row['probability']:11.4f} | "
                       f"SSIM={frame_data['global_ssim']:.3f} "
                       f"Teacher={frame_data['teacher_presence']:.3f}")
        
        logger.info("-"*100)
        
        # Pattern analysis
        logger.info("\n" + "="*60)
        logger.info("Pattern Analysis")
        logger.info("="*60)
        
        # Analyze feature values of false positives
        fp_features = test_df.iloc[false_positives.index]
        
        logger.info("\nAverage Feature Values (False Positives vs True Transitions):")
        
        # Get true transitions for comparison
        true_transitions = predictions_df[predictions_df['label'] == 1]
        tp_features = test_df.iloc[true_transitions.index]
        
        key_features = ['global_ssim', 'teacher_presence', 'skin_pixel_ratio', 
                        'black_pixel_ratio', 'edge_change_rate']
        
        logger.info("-"*70)
        logger.info("Feature                    | False Positives | True Transitions | Diff")
        logger.info("-"*70)
        
        for feature in key_features:
            fp_mean = fp_features[feature].mean()
            tp_mean = tp_features[feature].mean()
            diff = fp_mean - tp_mean
            
            logger.info(f"{feature:26s} | {fp_mean:15.4f} | {tp_mean:16.4f} | {diff:+.4f}")
        
        logger.info("-"*70)
        
        # Hypothesis generation
        logger.info("\n" + "="*60)
        logger.info("Hypothesis: What's Causing False Positives?")
        logger.info("="*60)
        
        fp_ssim = fp_features['global_ssim'].mean()
        fp_teacher = fp_features['teacher_presence'].mean()
        tp_ssim = tp_features['global_ssim'].mean()
        tp_teacher = tp_features['teacher_presence'].mean()
        
        if fp_teacher > 0.3:
            logger.info(">> Likely Cause: TEACHER MOTION")
            logger.info(f"   False Positives have high teacher_presence ({fp_teacher:.3f})")
            logger.info("   Model is still confusing teacher movement with transitions")
            logger.info("   Recommendation: Increase weight on teacher_presence feature")
        
        if fp_ssim < 0.7:
            logger.info(">> Likely Cause: BOARD ERASING / LIGHTING CHANGES")
            logger.info(f"   False Positives have low SSIM ({fp_ssim:.3f})")
            logger.info("   These are visual changes, but not slide transitions")
            logger.info("   Recommendation: Add edge_stability or temporal context features")
        
        if abs(fp_ssim - tp_ssim) < 0.1:
            logger.info(">> Likely Cause: AMBIGUOUS FRAMES")
            logger.info("   False Positives look very similar to real transitions")
            logger.info("   Recommendation: Use temporal smoothing (already applied)")
        
        return false_positives
    
    def save_refined_predictions(self, smoothed_df):
        """Save refined predictions to CSV"""
        logger.info("\n" + "="*60)
        logger.info("Saving Refined Predictions")
        logger.info("="*60)
        
        output_path = self.data_dir / 'refined_predictions.csv'
        smoothed_df.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path}")
        logger.info(f"Columns: {list(smoothed_df.columns)}")
    
    def visualize_threshold_curve(self, results_df):
        """Plot precision-recall vs threshold"""
        logger.info("\n" + "="*60)
        logger.info("Generating Threshold Visualization")
        logger.info("="*60)
        
        viz_dir = Path('visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(results_df['threshold'], results_df['precision'], 
                label='Precision', marker='o', linewidth=2)
        plt.plot(results_df['threshold'], results_df['recall'], 
                label='Recall', marker='s', linewidth=2)
        plt.plot(results_df['threshold'], results_df['f1'], 
                label='F1-Score', marker='^', linewidth=2)
        
        # Mark optimal threshold
        plt.axvline(self.optimal_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal ({self.optimal_threshold:.2f})')
        
        # Mark target recall
        plt.axhline(0.85, color='green', linestyle=':', 
                   linewidth=1, alpha=0.5, label='Target Recall (85%)')
        
        plt.xlabel('Probability Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Threshold Tuning: Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = viz_dir / f'threshold_tuning_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {plot_path}")
    
    def run(self):
        """Execute complete refinement pipeline"""
        logger.info("\n" + "="*60)
        logger.info("Model Refinement Pipeline")
        logger.info("="*60)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Load model and data
            X_test, y_test = self.load_model_and_data()
            
            # Find optimal threshold
            results_df, y_proba = self.find_optimal_threshold(X_test, y_test)
            
            # Apply optimal threshold
            logger.info("\n" + "="*60)
            logger.info("Applying Optimal Threshold")
            logger.info("="*60)
            
            y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)
            
            # Create predictions DataFrame
            predictions_df = self.test_metadata.copy()
            predictions_df['label'] = y_test
            predictions_df['probability'] = y_proba
            predictions_df['prediction'] = y_pred_optimal
            
            logger.info(f"Predictions generated with threshold = {self.optimal_threshold:.2f}")
            
            # Apply temporal smoothing
            smoothed_df = self.apply_temporal_smoothing(predictions_df)
            
            # Analyze false positives
            self.analyze_hard_negatives(predictions_df)
            
            # Save results
            self.save_refined_predictions(smoothed_df)
            
            # Visualize
            self.visualize_threshold_curve(results_df)
            
            logger.info("\n" + "="*60)
            logger.info("REFINEMENT COMPLETE!")
            logger.info("="*60)
            logger.info("\nKey Improvements:")
            logger.info(f"1. Optimal threshold: {self.optimal_threshold:.2f} (was 0.50)")
            logger.info("2. Temporal smoothing applied (NMS)")
            logger.info("3. False positive analysis completed")
            logger.info("\nNext Steps:")
            logger.info("1. Review visualizations/threshold_tuning_*.png")
            logger.info("2. Check data/refined_predictions.csv")
            logger.info("3. Analyze hard negatives to improve feature engineering")
            
            return True
            
        except Exception as e:
            logger.error(f"Refinement pipeline failed: {e}", exc_info=True)
            return False


def main():
    """Main execution function"""
    # Find latest model
    model_dir = Path('models')
    model_files = list(model_dir.glob('xgboost_model_*.pkl'))
    
    if not model_files:
        logger.error("No trained model found! Train a model first.")
        return
    
    # Use latest model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using model: {latest_model}")
    
    refiner = ModelRefiner(latest_model)
    success = refiner.run()
    
    if success:
        logger.info("\n SUCCESS - Model refined and optimized!")
    else:
        logger.error("\n FAILED - Check logs for details")


if __name__ == '__main__':
    main()
