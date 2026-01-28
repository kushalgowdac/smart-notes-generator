"""
XGBoost Transition Classifier with Teacher Noise Handling
=========================================================
Trains an XGBoost model to distinguish slide transitions from teacher motion.
Uses teacher_presence as a "shield" to ignore teacher movement.

Key Logic:
- IF global_ssim drops AND teacher_presence is LOW  = Transition
- IF global_ssim drops AND teacher_presence is HIGH = Teacher Motion (Ignore)

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
    roc_auc_score,
    roc_curve
)
import xgboost as xgb
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TransitionClassifier:
    """XGBoost classifier for slide transition detection"""
    
    def __init__(self, data_dir='data', model_dir='models'):
        """
        Initialize the classifier
        
        Args:
            data_dir: Directory containing train/test datasets
            model_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.feature_columns = None
        self.scale_pos_weight = None
        
        logger.info("Initialized TransitionClassifier")
        logger.info(f"  Data Directory: {self.data_dir}")
        logger.info(f"  Model Directory: {self.model_dir}")
    
    def load_datasets(self):
        """
        Load training and test datasets
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test, train_videos, test_videos)
        """
        logger.info("\n" + "="*60)
        logger.info("Loading Datasets")
        logger.info("="*60)
        
        # Load training set
        train_path = self.data_dir / 'train_dataset.csv'
        logger.info(f"Loading training set: {train_path}")
        train_df = pd.read_csv(train_path)
        logger.info(f"  Shape: {train_df.shape}")
        logger.info(f"  Videos: {train_df['video_id'].nunique()}")
        
        # Load test set
        test_path = self.data_dir / 'test_dataset.csv'
        logger.info(f"\nLoading test set: {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"  Shape: {test_df.shape}")
        logger.info(f"  Videos: {test_df['video_id'].nunique()}")
        
        # Feature columns (exclude video_id, frame_index, timestamp_seconds, label)
        exclude_cols = ['video_id', 'frame_index', 'timestamp_seconds', 'label']
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        logger.info(f"\nFeature columns ({len(self.feature_columns)}):")
        for i, col in enumerate(self.feature_columns, 1):
            logger.info(f"  [{i:2d}] {col}")
        
        # Separate features and labels
        X_train = train_df[self.feature_columns].values
        y_train = train_df['label'].values
        X_test = test_df[self.feature_columns].values
        y_test = test_df['label'].values
        
        # Store video IDs for analysis
        train_videos = train_df['video_id'].values
        test_videos = test_df['video_id'].values
        
        # Calculate class distribution
        logger.info("\n" + "="*60)
        logger.info("Class Distribution")
        logger.info("="*60)
        
        train_0 = (y_train == 0).sum()
        train_1 = (y_train == 1).sum()
        test_0 = (y_test == 0).sum()
        test_1 = (y_test == 1).sum()
        
        logger.info(f"\nTraining Set:")
        logger.info(f"  Class 0 (Non-Transition): {train_0:,} ({train_0/len(y_train)*100:.2f}%)")
        logger.info(f"  Class 1 (Transition):     {train_1:,} ({train_1/len(y_train)*100:.2f}%)")
        logger.info(f"  Imbalance Ratio: {train_0/train_1:.2f}:1")
        
        logger.info(f"\nTest Set:")
        logger.info(f"  Class 0 (Non-Transition): {test_0:,} ({test_0/len(y_test)*100:.2f}%)")
        logger.info(f"  Class 1 (Transition):     {test_1:,} ({test_1/len(y_test)*100:.2f}%)")
        logger.info(f"  Imbalance Ratio: {test_0/test_1:.2f}:1")
        
        # Calculate scale_pos_weight for XGBoost
        self.scale_pos_weight = train_0 / train_1
        logger.info(f"\nCalculated scale_pos_weight: {self.scale_pos_weight:.2f}")
        logger.info("  (This tells XGBoost to weight positive class higher)")
        
        return X_train, y_train, X_test, y_test, train_videos, test_videos
    
    def train_model(self, X_train, y_train):
        """
        Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("\n" + "="*60)
        logger.info("Training XGBoost Model")
        logger.info("="*60)
        
        # XGBoost parameters optimized for imbalanced classification
        params = {
            'max_depth': 6,                      # Tree depth
            'learning_rate': 0.1,                # Step size shrinkage
            'n_estimators': 200,                 # Number of trees
            'objective': 'binary:logistic',      # Binary classification
            'scale_pos_weight': self.scale_pos_weight,  # Handle imbalance
            'subsample': 0.8,                    # Row sampling
            'colsample_bytree': 0.8,             # Column sampling
            'min_child_weight': 3,               # Minimum sum of instance weight
            'gamma': 0.1,                        # Minimum loss reduction
            'reg_alpha': 0.1,                    # L1 regularization
            'reg_lambda': 1.0,                   # L2 regularization
            'random_state': 42,
            'n_jobs': -1,                        # Use all cores
            'eval_metric': 'logloss'
        }
        
        logger.info("Model Parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # Train model
        logger.info("\nTraining model...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, verbose=False)
        
        logger.info("Training complete!")
    
    def evaluate_model(self, X_test, y_test, test_videos):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            test_videos: Test video IDs
        """
        logger.info("\n" + "="*60)
        logger.info("Model Evaluation on Test Set")
        logger.info("="*60)
        
        # Predictions
        logger.info("\nGenerating predictions...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("\n" + "="*60)
        logger.info("Overall Performance Metrics")
        logger.info("="*60)
        logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Precision: {precision:.4f} (How many predicted transitions are correct)")
        logger.info(f"Recall:    {recall:.4f} (How many actual transitions we caught)")
        logger.info(f"F1-Score:  {f1:.4f} (Harmonic mean of precision & recall)")
        logger.info(f"ROC-AUC:   {auc:.4f} (Overall discriminative ability)")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\n" + "="*60)
        logger.info("Confusion Matrix")
        logger.info("="*60)
        logger.info(f"                 Predicted: 0    Predicted: 1")
        logger.info(f"Actual: 0 (Non-Trans)  {cm[0,0]:6d}         {cm[0,1]:6d}")
        logger.info(f"Actual: 1 (Transition) {cm[1,0]:6d}         {cm[1,1]:6d}")
        logger.info("")
        logger.info(f"True Negatives (TN):  {cm[0,0]:,} (Correctly identified non-transitions)")
        logger.info(f"False Positives (FP): {cm[0,1]:,} (Wrongly predicted as transition)")
        logger.info(f"False Negatives (FN): {cm[1,0]:,} (Missed transitions)")
        logger.info(f"True Positives (TP):  {cm[1,1]:,} (Correctly caught transitions)")
        
        # Per-video performance
        logger.info("\n" + "="*60)
        logger.info("Per-Video Performance (Test Set)")
        logger.info("="*60)
        
        unique_videos = np.unique(test_videos)
        for video_id in unique_videos:
            mask = test_videos == video_id
            y_true_video = y_test[mask]
            y_pred_video = y_pred[mask]
            
            acc = accuracy_score(y_true_video, y_pred_video)
            prec = precision_score(y_true_video, y_pred_video, zero_division=0)
            rec = recall_score(y_true_video, y_pred_video, zero_division=0)
            f1_v = f1_score(y_true_video, y_pred_video, zero_division=0)
            
            logger.info(f"\n{video_id}:")
            logger.info(f"  Frames: {len(y_true_video)}")
            logger.info(f"  Transitions: {(y_true_video==1).sum()}")
            logger.info(f"  Accuracy:  {acc:.4f}")
            logger.info(f"  Precision: {prec:.4f}")
            logger.info(f"  Recall:    {rec:.4f}")
            logger.info(f"  F1-Score:  {f1_v:.4f}")
        
        # Classification report
        logger.info("\n" + "="*60)
        logger.info("Detailed Classification Report")
        logger.info("="*60)
        logger.info("\n" + classification_report(y_test, y_pred, 
                                                 target_names=['Non-Transition', 'Transition'],
                                                 digits=4))
        
        return y_pred, y_pred_proba, cm
    
    def analyze_feature_importance(self):
        """
        Analyze and display feature importance
        Focus on teacher_presence and skin_pixel_ratio
        """
        logger.info("\n" + "="*60)
        logger.info("Feature Importance Analysis")
        logger.info("="*60)
        logger.info("(Higher importance = More useful for detecting transitions)")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 15 Most Important Features:")
        logger.info("-" * 60)
        for i, row in importance_df.head(15).iterrows():
            bar = '█' * int(row['importance'] * 100)
            logger.info(f"{row['feature']:30s} {row['importance']:.4f} {bar}")
        
        # Highlight teacher-related features
        logger.info("\n" + "="*60)
        logger.info("Teacher Noise Handling Features")
        logger.info("="*60)
        
        teacher_features = ['teacher_presence', 'skin_pixel_ratio', 'black_pixel_ratio']
        for feature in teacher_features:
            if feature in importance_df['feature'].values:
                imp = importance_df[importance_df['feature'] == feature]['importance'].values[0]
                rank = importance_df[importance_df['feature'] == feature].index[0] + 1
                logger.info(f"{feature:25s} | Importance: {imp:.4f} | Rank: {rank}/22")
        
        # Check if teacher_presence is helping
        teacher_imp = importance_df[importance_df['feature'] == 'teacher_presence']['importance'].values[0]
        ssim_imp = importance_df[importance_df['feature'] == 'global_ssim']['importance'].values[0]
        
        logger.info("\n" + "="*60)
        logger.info("Is teacher_presence Helping?")
        logger.info("="*60)
        
        if teacher_imp > 0.02:
            logger.info(f"✓ YES! teacher_presence has importance {teacher_imp:.4f}")
            logger.info("  The model is using it to distinguish teacher motion from transitions")
        else:
            logger.info(f"✗ Weak signal. teacher_presence importance is only {teacher_imp:.4f}")
            logger.info("  The model may not be leveraging it effectively")
        
        logger.info(f"\nComparison:")
        logger.info(f"  global_ssim importance:     {ssim_imp:.4f}")
        logger.info(f"  teacher_presence importance: {teacher_imp:.4f}")
        logger.info(f"  Ratio: {teacher_imp/ssim_imp:.2f}x")
        
        return importance_df
    
    def save_model(self):
        """Save trained model and metadata"""
        logger.info("\n" + "="*60)
        logger.info("Saving Model")
        logger.info("="*60)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = self.model_dir / f'xgboost_transition_classifier_{timestamp}.json'
        self.model.save_model(model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save feature columns
        features_path = self.model_dir / f'feature_columns_{timestamp}.txt'
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        logger.info(f"Feature columns saved: {features_path}")
        
        # Save model using joblib (for easier loading)
        joblib_path = self.model_dir / f'xgboost_model_{timestamp}.pkl'
        joblib.dump(self.model, joblib_path)
        logger.info(f"Model (joblib) saved: {joblib_path}")
        
        logger.info("\nModel artifacts saved successfully!")
    
    def save_visualizations(self, cm, importance_df):
        """
        Save confusion matrix and feature importance plots
        
        Args:
            cm: Confusion matrix
            importance_df: Feature importance DataFrame
        """
        logger.info("\n" + "="*60)
        logger.info("Generating Visualizations")
        logger.info("="*60)
        
        # Create output directory
        viz_dir = Path('visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Transition', 'Transition'],
                    yticklabels=['Non-Transition', 'Transition'])
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        cm_path = viz_dir / f'confusion_matrix_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved: {cm_path}")
        
        # 2. Feature Importance (Top 15)
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        colors = ['red' if 'teacher' in f or 'skin' in f or 'black' in f 
                  else 'steelblue' for f in top_features['feature']]
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 15 Feature Importance\n(Red = Teacher-related features)', 
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        importance_path = viz_dir / f'feature_importance_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance saved: {importance_path}")
        
        logger.info("\nVisualizations saved successfully!")
    
    def run(self):
        """Execute complete training pipeline"""
        logger.info("\n" + "="*60)
        logger.info("XGBoost Training Pipeline - Pilot Run")
        logger.info("="*60)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\nGoal: Train model to distinguish transitions from teacher motion")
        logger.info("Key: teacher_presence as 'shield' to ignore teacher movement")
        
        try:
            # Load datasets
            X_train, y_train, X_test, y_test, train_videos, test_videos = self.load_datasets()
            
            # Train model
            self.train_model(X_train, y_train)
            
            # Evaluate model
            y_pred, y_pred_proba, cm = self.evaluate_model(X_test, y_test, test_videos)
            
            # Analyze feature importance
            importance_df = self.analyze_feature_importance()
            
            # Save model
            self.save_model()
            
            # Save visualizations
            self.save_visualizations(cm, importance_df)
            
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETE!")
            logger.info("="*60)
            logger.info("\nNext Steps:")
            logger.info("1. Check visualizations/ folder for plots")
            logger.info("2. Review feature importance - is teacher_presence helping?")
            logger.info("3. If results good, deploy model for real-time prediction")
            logger.info("4. If results poor, consider feature engineering or hyperparameter tuning")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            return False


def main():
    """Main execution function"""
    classifier = TransitionClassifier()
    success = classifier.run()
    
    if success:
        logger.info("\n✓ SUCCESS - Model trained and ready!")
    else:
        logger.error("\n✗ FAILED - Check logs for details")


if __name__ == '__main__':
    main()
