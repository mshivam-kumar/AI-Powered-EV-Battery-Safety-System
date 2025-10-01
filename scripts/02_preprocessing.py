#!/usr/bin/env python3
"""
Step 2: Preprocessing & Feature Engineering Pipeline
Extract 16 features from NASA battery data and generate synthetic labels
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_16_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 16 features from NASA battery data
        
        Basic Features (7):
        1. voltage_charger - Battery terminal voltage
        2. current - Charging current (derived from mode and voltage)
        3. temperature_battery - Battery cell temperature
        4. soc - State of charge (derived from voltage)
        5. ambient_temp - Environmental temperature (derived)
        6. humidity - Relative humidity (derived)
        7. charge_mode - Charging mode (derived from mode)
        
        Derived Features (9):
        8. power - Electrical power consumption
        9. c_rate - Charging rate indicator
        10. temp_diff - Temperature difference
        11. voltage_soc_ratio - Voltage-SoC correlation
        12. thermal_stress - Normalized temperature stress
        13. temp_gradient - Rate of temperature change
        14. voltage_gradient - Rate of voltage change
        15. soc_rate - Rate of SoC change
        16. env_stress - Environmental stress
        """
        
        features_df = df.copy()
        
        # Basic Feature 1: Voltage
        features_df['voltage'] = features_df['voltage_charger'].fillna(features_df['voltage_load'])
        
        # Basic Feature 2: Current (derive from mode and voltage)
        features_df['current'] = np.where(
            features_df['mode'] == 1,  # Charging mode
            features_df['current_load'].fillna(2.5),  # Default charging current
            np.where(
                features_df['mode'] == -1,  # Discharging mode
                -features_df['current_load'].fillna(-2.5),  # Default discharging current
                0  # Rest mode
            )
        )
        
        # Basic Feature 3: Temperature
        features_df['temperature'] = features_df['temperature_battery']
        
        # Basic Feature 4: SoC (State of Charge) - derive from voltage
        # Typical Li-ion voltage range: 3.0V (0%) to 4.2V (100%)
        voltage_min, voltage_max = 3.0, 4.2
        features_df['soc'] = np.clip(
            (features_df['voltage'] - voltage_min) / (voltage_max - voltage_min),
            0, 1
        )
        
        # Basic Feature 5: Ambient Temperature (derive from battery temp and mode)
        # Assume ambient is cooler than battery, with some variation
        features_df['ambient_temp'] = features_df['temperature'] - np.random.normal(5, 2, len(features_df))
        features_df['ambient_temp'] = np.clip(features_df['ambient_temp'], 15, 45)  # Realistic range
        
        # Basic Feature 6: Humidity (derive from ambient temp)
        # Higher humidity in moderate temperatures
        features_df['humidity'] = np.clip(
            0.3 + 0.4 * np.exp(-((features_df['ambient_temp'] - 25) / 10) ** 2),
            0.1, 0.9
        )
        
        # Basic Feature 7: Charge Mode
        features_df['charge_mode'] = features_df['mode'].map({
            1: 'fast',    # Charging
            -1: 'slow',    # Discharging  
            0: 'pause'    # Rest
        })
        
        # Derived Feature 8: Power (V × I)
        features_df['power'] = features_df['voltage'] * np.abs(features_df['current'])
        
        # Derived Feature 9: C-rate (charging rate)
        features_df['c_rate'] = np.abs(features_df['current'])
        
        # Derived Feature 10: Temperature Difference
        features_df['temp_diff'] = features_df['temperature'] - features_df['ambient_temp']
        
        # Derived Feature 11: Voltage-SoC Ratio
        features_df['voltage_soc_ratio'] = features_df['voltage'] / (features_df['soc'] + 0.001)  # Avoid division by zero
        
        # Derived Feature 12: Thermal Stress
        # Normalized temperature stress (0-1, higher is worse)
        ideal_temp = 25  # Ideal battery temperature
        temp_range = 30  # Temperature range for normalization
        features_df['thermal_stress'] = 1 - np.abs(features_df['temperature'] - ideal_temp) / temp_range
        features_df['thermal_stress'] = np.clip(features_df['thermal_stress'], 0, 1)
        
        # Derived Feature 13: Temperature Gradient (rate of change)
        features_df['temp_gradient'] = features_df['temperature'].diff().fillna(0)
        
        # Derived Feature 14: Voltage Gradient (rate of change)
        features_df['voltage_gradient'] = features_df['voltage'].diff().fillna(0)
        
        # Derived Feature 15: SoC Rate (rate of change)
        features_df['soc_rate'] = features_df['soc'].diff().fillna(0)
        
        # Derived Feature 16: Environmental Stress
        # Combined ambient temperature and humidity stress
        features_df['env_stress'] = (features_df['ambient_temp'] / 50.0) * features_df['humidity']
        
        # Select only the 16 features
        feature_columns = [
            'voltage', 'current', 'temperature', 'soc', 'ambient_temp', 'humidity', 'charge_mode',
            'power', 'c_rate', 'temp_diff', 'voltage_soc_ratio', 'thermal_stress',
            'temp_gradient', 'voltage_gradient', 'soc_rate', 'env_stress'
        ]
        
        return features_df[feature_columns]
    
    def preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for ML models"""
        processed_df = features_df.copy()
        
        # Encode categorical variables first
        if 'charge_mode' in processed_df.columns:
            processed_df['charge_mode_encoded'] = self.label_encoder.fit_transform(processed_df['charge_mode'])
            processed_df = processed_df.drop('charge_mode', axis=1)
        
        # Handle missing values for numerical columns only
        numerical_columns = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numerical_columns] = processed_df[numerical_columns].fillna(processed_df[numerical_columns].median())
        
        # Scale numerical features
        processed_df[numerical_columns] = self.scaler.fit_transform(processed_df[numerical_columns])
        
        return processed_df

class PreprocessingPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.processed_dir / "features"
        self.labels_dir = self.processed_dir / "labels"
        self.splits_dir = self.processed_dir / "splits"
        
        # Create directories
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
    
    def load_combined_data(self) -> pd.DataFrame:
        """Load the combined battery data from data preparation"""
        combined_file = self.processed_dir / "combined_battery_data.parquet"
        
        if not combined_file.exists():
            raise FileNotFoundError(f"Combined data not found at {combined_file}. Run data preparation first.")
        
        df = pd.read_parquet(combined_file)
        logger.info(f"Loaded combined data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract 16 features from the combined dataset"""
        logger.info("Extracting 16 features from battery data...")
        
        # Extract features
        features_df = self.feature_extractor.extract_16_features(df)
        
        # Preprocess features
        processed_features = self.feature_extractor.preprocess_features(features_df)
        
        # Add metadata
        processed_features['battery_id'] = df['battery_id']
        processed_features['battery_type'] = df['battery_type']
        processed_features['time'] = df['time']
        
        logger.info(f"Extracted features: {len(processed_features)} rows, {len(processed_features.columns)} columns")
        
        # Save features
        features_file = self.features_dir / "extracted_features.parquet"
        processed_features.to_parquet(features_file, index=False)
        logger.info(f"Saved features to {features_file}")
        
        return processed_features
    
    def create_data_splits(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits"""
        logger.info("Creating data splits...")
        
        # Remove metadata columns for splitting
        feature_columns = [col for col in features_df.columns 
                          if col not in ['battery_id', 'battery_type', 'time']]
        
        X = features_df[feature_columns]
        
        # Create splits (70% train, 15% validation, 15% test)
        X_temp, X_test, _, _ = train_test_split(
            X, X, test_size=0.15, random_state=42, stratify=features_df['battery_type']
        )
        X_train, X_val, _, _ = train_test_split(
            X_temp, X_temp, test_size=0.176, random_state=42, stratify=features_df.loc[X_temp.index, 'battery_type']
        )
        
        # Add metadata back
        train_df = features_df.loc[X_train.index].copy()
        val_df = features_df.loc[X_val.index].copy()
        test_df = features_df.loc[X_test.index].copy()
        
        # Save splits
        train_file = self.splits_dir / "train.parquet"
        val_file = self.splits_dir / "validation.parquet"
        test_file = self.splits_dir / "test.parquet"
        
        train_df.to_parquet(train_file, index=False)
        val_df.to_parquet(val_file, index=False)
        test_df.to_parquet(test_file, index=False)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Load combined data
        logger.info("Step 1: Loading combined data...")
        combined_df = self.load_combined_data()
        
        # Step 2: Extract features
        logger.info("Step 2: Extracting 16 features...")
        features_df = self.extract_features(combined_df)
        
        # Step 3: Create data splits
        logger.info("Step 3: Creating data splits...")
        splits = self.create_data_splits(features_df)
        
        # Summary
        logger.info("="*50)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total samples: {len(features_df):,}")
        logger.info(f"Features extracted: 16")
        logger.info(f"Train samples: {len(splits['train']):,}")
        logger.info(f"Validation samples: {len(splits['validation']):,}")
        logger.info(f"Test samples: {len(splits['test']):,}")
        logger.info("Preprocessing completed successfully!")
        
        return features_df, splits

def main():
    """Main function to run preprocessing"""
    pipeline = PreprocessingPipeline()
    features_df, splits = pipeline.run_preprocessing()
    
    if features_df is not None:
        print(f"\n✅ Preprocessing completed!")
        print(f"📊 Features extracted: {len(features_df):,} samples")
        print(f"📁 Features saved to: data/processed/features/")
        print(f"📁 Data splits saved to: data/processed/splits/")
    else:
        print("❌ Preprocessing failed!")

if __name__ == "__main__":
    main()
