"""
Merge Labeled CSVs and Split into Train/Test Sets
==================================================
Combines all 19 labeled feature CSVs into a master dataset,
then splits by video ID for ML training.

Author: Smart Notes Generator Team
Date: January 26, 2026
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_merge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetMerger:
    """Merge labeled CSVs and split into train/test sets"""
    
    def __init__(self, input_dir='data/output', output_dir='data'):
        """
        Initialize the merger
        
        Args:
            input_dir: Directory containing labeled CSV files
            output_dir: Directory to save merged datasets
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test set video IDs (4 videos for final evaluation)
        self.test_videos = [
            'chemistry_01_english',
            'toc_1',
            'mathematics_03_english',
            'chemistry_10_english'
        ]
        
        logger.info("Initialized DatasetMerger")
        logger.info(f"  Input Directory: {self.input_dir}")
        logger.info(f"  Output Directory: {self.output_dir}")
        logger.info(f"  Test Videos: {', '.join(self.test_videos)}")
    
    def load_all_csvs(self):
        """
        Load all labeled CSV files
        
        Returns:
            DataFrame: Combined dataset from all videos
        """
        logger.info("\n" + "="*60)
        logger.info("Loading All Labeled CSVs")
        logger.info("="*60)
        
        csv_files = sorted(self.input_dir.glob('*_features.csv'))
        
        if not csv_files:
            logger.error(f"No CSV files found in {self.input_dir}")
            return None
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dataframes = []
        total_frames = 0
        total_transitions = 0
        
        for csv_file in csv_files:
            video_name = csv_file.stem.replace('_features', '')
            
            try:
                df = pd.read_csv(csv_file)
                
                # Validate columns
                if 'label' not in df.columns:
                    logger.warning(f"Skipping {video_name}: No 'label' column found")
                    continue
                
                frames = len(df)
                transitions = (df['label'] == 1).sum()
                
                total_frames += frames
                total_transitions += transitions
                
                logger.info(f"  [{len(dataframes)+1:2d}] {video_name:30s} | "
                           f"Frames: {frames:5d} | Transitions: {transitions:4d} | "
                           f"Ratio: {transitions/frames*100:5.2f}%")
                
                dataframes.append(df)
                
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {e}")
                continue
        
        if not dataframes:
            logger.error("No valid CSV files loaded")
            return None
        
        # Concatenate all dataframes
        logger.info("\nConcatenating all dataframes...")
        master_df = pd.concat(dataframes, ignore_index=True)
        
        logger.info("\n" + "="*60)
        logger.info("Master Dataset Summary")
        logger.info("="*60)
        logger.info(f"Total Videos: {len(dataframes)}")
        logger.info(f"Total Frames: {total_frames:,}")
        logger.info(f"Total Transitions: {total_transitions:,}")
        logger.info(f"Average Transition Ratio: {total_transitions/total_frames*100:.2f}%")
        logger.info(f"Dataset Shape: {master_df.shape}")
        logger.info(f"Columns: {list(master_df.columns)}")
        
        return master_df
    
    def split_train_test(self, master_df):
        """
        Split dataset into train and test sets by video ID
        
        Args:
            master_df: Combined dataset
            
        Returns:
            tuple: (train_df, test_df)
        """
        logger.info("\n" + "="*60)
        logger.info("Splitting Train/Test Sets")
        logger.info("="*60)
        
        # Split by video_id
        test_mask = master_df['video_id'].isin(self.test_videos)
        test_df = master_df[test_mask].copy()
        train_df = master_df[~test_mask].copy()
        
        # Training set stats
        logger.info("\nTraining Set:")
        train_videos = train_df['video_id'].unique()
        logger.info(f"  Videos: {len(train_videos)}")
        logger.info(f"  Video IDs: {', '.join(sorted(train_videos))}")
        logger.info(f"  Total Frames: {len(train_df):,}")
        logger.info(f"  Transitions: {(train_df['label']==1).sum():,}")
        logger.info(f"  Transition Ratio: {(train_df['label']==1).sum()/len(train_df)*100:.2f}%")
        
        # Test set stats
        logger.info("\nTest Set:")
        test_videos = test_df['video_id'].unique()
        logger.info(f"  Videos: {len(test_videos)}")
        logger.info(f"  Video IDs: {', '.join(sorted(test_videos))}")
        logger.info(f"  Total Frames: {len(test_df):,}")
        logger.info(f"  Transitions: {(test_df['label']==1).sum():,}")
        logger.info(f"  Transition Ratio: {(test_df['label']==1).sum()/len(test_df)*100:.2f}%")
        
        # Verify all test videos are present
        missing_test = set(self.test_videos) - set(test_videos)
        if missing_test:
            logger.warning(f"Missing test videos: {', '.join(missing_test)}")
        
        return train_df, test_df
    
    def save_datasets(self, master_df, train_df, test_df):
        """
        Save master, train, and test datasets to CSV files
        
        Args:
            master_df: Complete combined dataset
            train_df: Training set
            test_df: Test set
        """
        logger.info("\n" + "="*60)
        logger.info("Saving Datasets")
        logger.info("="*60)
        
        # Save master dataset
        master_path = self.output_dir / 'master_dataset.csv'
        master_df.to_csv(master_path, index=False)
        logger.info(f"Saved master dataset: {master_path}")
        logger.info(f"  Shape: {master_df.shape}")
        logger.info(f"  Size: {master_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Save training set
        train_path = self.output_dir / 'train_dataset.csv'
        train_df.to_csv(train_path, index=False)
        logger.info(f"\nSaved training set: {train_path}")
        logger.info(f"  Shape: {train_df.shape}")
        logger.info(f"  Size: {train_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Save test set
        test_path = self.output_dir / 'test_dataset.csv'
        test_df.to_csv(test_path, index=False)
        logger.info(f"\nSaved test set: {test_path}")
        logger.info(f"  Shape: {test_df.shape}")
        logger.info(f"  Size: {test_path.stat().st_size / (1024*1024):.2f} MB")
        
        logger.info("\n" + "="*60)
        logger.info("Dataset Merge Complete!")
        logger.info("="*60)
    
    def run(self):
        """Execute the complete merge and split pipeline"""
        logger.info("\n" + "="*60)
        logger.info("Dataset Merge and Split Pipeline")
        logger.info("="*60)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Load all CSVs
            master_df = self.load_all_csvs()
            if master_df is None:
                logger.error("Failed to load CSVs")
                return False
            
            # Split into train/test
            train_df, test_df = self.split_train_test(master_df)
            
            # Save all datasets
            self.save_datasets(master_df, train_df, test_df)
            
            logger.info("\nPipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False


def main():
    """Main execution function"""
    merger = DatasetMerger()
    success = merger.run()
    
    if success:
        logger.info("\n SUCCESS - All datasets ready for ML training!")
    else:
        logger.error("\n FAILED - Check logs for details")


if __name__ == '__main__':
    main()
