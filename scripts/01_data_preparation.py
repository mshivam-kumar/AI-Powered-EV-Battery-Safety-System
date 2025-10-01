#!/usr/bin/env python3
"""
Step 1: Data Preparation Pipeline
Load and validate NASA Battery Alternative Dataset from all subdirectories
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "battery_alt_dataset"
        self.processed_dir = self.data_dir / "processed"
        
    def load_all_battery_data(self) -> Dict[str, pd.DataFrame]:
        """Load all battery CSV files from all subdirectories"""
        battery_data = {}
        
        # Define subdirectories and their purposes
        subdirs = {
            'regular_alt_batteries': 'Constant load cycling',
            'recommissioned_batteries': 'Variable load cycling', 
            'second_life_batteries': 'Second life cycling'
        }
        
        for subdir, description in subdirs.items():
            subdir_path = self.raw_dir / subdir
            logger.info(f"Loading data from {subdir}: {description}")
            
            if subdir_path.exists():
                csv_files = list(subdir_path.glob("*.csv"))
                logger.info(f"Found {len(csv_files)} CSV files in {subdir}")
                
                for csv_file in csv_files:
                    try:
                        # Load CSV file with proper data types
                        df = pd.read_csv(csv_file, low_memory=False)
                        
                        # Convert numeric columns to proper types
                        numeric_columns = ['time', 'mode', 'voltage_charger', 'temperature_battery', 
                                         'voltage_load', 'current_load', 'temperature_mosfet', 
                                         'temperature_resistor', 'mission_type']
                        
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Add metadata columns
                        df['battery_id'] = csv_file.stem
                        df['battery_type'] = subdir
                        df['battery_description'] = description
                        
                        # Limit sample size to avoid memory issues (take every 10th row)
                        if len(df) > 100000:
                            df = df.iloc[::10]  # Take every 10th row
                            logger.info(f"  Sampled {len(df)} rows from {csv_file.name}")
                        
                        # Store with descriptive key
                        key = f"{subdir}_{csv_file.stem}"
                        battery_data[key] = df
                        
                        logger.info(f"Loaded {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
                        
                    except Exception as e:
                        logger.error(f"Error loading {csv_file}: {e}")
            else:
                logger.warning(f"Directory {subdir} not found")
        
        return battery_data
    
    def validate_data_quality(self, battery_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Validate data quality across all battery datasets"""
        quality_report = {
            'total_batteries': len(battery_data),
            'total_rows': 0,
            'quality_issues': [],
            'column_analysis': {},
            'battery_summary': {}
        }
        
        # Expected columns from NASA dataset
        expected_columns = [
            'start_time', 'time', 'mode', 'voltage_charger', 'temperature_battery',
            'voltage_load', 'current_load', 'temperature_mosfet', 'temperature_resistor', 'mission_type'
        ]
        
        for battery_id, df in battery_data.items():
            battery_summary = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict(),
                'time_range': {
                    'start': df['time'].min() if 'time' in df.columns else None,
                    'end': df['time'].max() if 'time' in df.columns else None
                }
            }
            
            quality_report['battery_summary'][battery_id] = battery_summary
            quality_report['total_rows'] += len(df)
            
            # Check for missing expected columns
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                quality_report['quality_issues'].append(f"{battery_id}: Missing columns {missing_cols}")
            
            # Check for excessive missing values
            high_missing = df.isnull().sum() / len(df) > 0.5
            if high_missing.any():
                quality_report['quality_issues'].append(f"{battery_id}: High missing values in {high_missing[high_missing].index.tolist()}")
        
        # Column analysis across all datasets
        all_columns = set()
        for df in battery_data.values():
            all_columns.update(df.columns)
        
        quality_report['column_analysis'] = {
            'all_columns': list(all_columns),
            'expected_columns': expected_columns,
            'missing_expected': set(expected_columns) - all_columns,
            'extra_columns': all_columns - set(expected_columns)
        }
        
        return quality_report
    
    def save_processed_data(self, battery_data: Dict[str, pd.DataFrame], quality_report: Dict[str, any]):
        """Save processed data and quality report"""
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual battery data
        for battery_id, df in battery_data.items():
            output_file = self.processed_dir / f"{battery_id}.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"Saved {battery_id} to {output_file}")
        
        # Save quality report
        quality_file = self.processed_dir / "data_quality_report.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        logger.info(f"Saved quality report to {quality_file}")
        
        # Save combined dataset
        combined_df = pd.concat(battery_data.values(), ignore_index=True)
        combined_file = self.processed_dir / "combined_battery_data.parquet"
        combined_df.to_parquet(combined_file, index=False)
        logger.info(f"Saved combined dataset to {combined_file}: {len(combined_df)} rows")
        
        return combined_df
    
    def run_data_preparation(self):
        """Run complete data preparation pipeline"""
        logger.info("Starting data preparation pipeline...")
        
        # Step 1: Load all battery data
        logger.info("Step 1: Loading all battery datasets...")
        battery_data = self.load_all_battery_data()
        
        if not battery_data:
            logger.error("No battery data loaded!")
            return None
        
        # Step 2: Validate data quality
        logger.info("Step 2: Validating data quality...")
        quality_report = self.validate_data_quality(battery_data)
        
        # Step 3: Save processed data
        logger.info("Step 3: Saving processed data...")
        combined_df = self.save_processed_data(battery_data, quality_report)
        
        # Summary
        logger.info("="*50)
        logger.info("DATA PREPARATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total batteries loaded: {quality_report['total_batteries']}")
        logger.info(f"Total rows: {quality_report['total_rows']:,}")
        logger.info(f"Quality issues: {len(quality_report['quality_issues'])}")
        
        if quality_report['quality_issues']:
            logger.warning("Quality issues found:")
            for issue in quality_report['quality_issues']:
                logger.warning(f"  - {issue}")
        
        logger.info("Data preparation completed successfully!")
        return combined_df

def main():
    """Main function to run data preparation"""
    data_prep = DataPreparation()
    combined_df = data_prep.run_data_preparation()
    
    if combined_df is not None:
        print(f"\n✅ Data preparation completed!")
        print(f"📊 Combined dataset: {len(combined_df):,} rows")
        print(f"📁 Processed data saved to: data/processed/")
    else:
        print("❌ Data preparation failed!")

if __name__ == "__main__":
    main()
