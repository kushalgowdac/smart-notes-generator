"""
Quick launcher for production note maker
Uses all_predictions.csv with all 19 videos
"""

import sys
sys.path.insert(0, 'src')
from production_note_maker import ProductionNoteMaker

# Run full pipeline
note_maker = ProductionNoteMaker(predictions_csv='data/all_predictions.csv')
note_maker.run_pipeline()
