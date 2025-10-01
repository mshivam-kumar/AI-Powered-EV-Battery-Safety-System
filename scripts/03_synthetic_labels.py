#!/usr/bin/env python3
"""
Step 3: Synthetic Label Generation Pipeline
Generate synthetic anomaly labels using multiple unsupervised methods
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticLabelGenerator:
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.methods = {
            'isolation_forest': IsolationForest(
                n_estimators=100, 
                contamination=contamination, 
                random_state=42
            ),
            'one_class_svm': OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='scale'
            ),
            'local_outlier_factor': LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
        }
        
    def generate_labels(self, X: np.ndarray, method: str) -> np.ndarray:
        """Generate synthetic labels using specified method"""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        model = self.methods[method]
        labels = model.fit_predict(X)
        
        # Convert to binary (1 = normal, 0 = anomaly)
        binary_labels = (labels == 1).astype(int)
        
        return binary_labels
    
    def generate_all_labels(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate labels using all methods"""
        all_labels = {}
        
        for method_name in self.methods.keys():
            logger.info(f"Generating labels using {method_name}...")
            try:
                labels = self.generate_labels(X, method_name)
                all_labels[method_name] = labels
                
                anomaly_count = np.sum(labels == 0)
                normal_count = np.sum(labels == 1)
                logger.info(f"  {method_name}: {anomaly_count} anomalies, {normal_count} normal")
                
            except Exception as e:
                logger.error(f"Error with {method_name}: {e}")
                all_labels[method_name] = None
        
        return all_labels
    
    def generate_consensus_labels(self, all_labels: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate consensus labels using majority voting"""
        logger.info("Generating consensus labels using majority voting...")
        
        # Filter out None values
        valid_labels = {k: v for k, v in all_labels.items() if v is not None}
        
        if not valid_labels:
            raise ValueError("No valid labels generated!")
        
        # Convert to array for voting
        label_arrays = np.array(list(valid_labels.values()))
        
        # Majority voting
        consensus_labels = []
        for i in range(len(label_arrays[0])):
            votes = label_arrays[:, i]
            # If majority vote for anomaly (0), mark as anomaly
            if np.sum(votes == 0) > len(votes) // 2:
                consensus_labels.append(0)  # Anomaly
            else:
                consensus_labels.append(1)  # Normal
        
        consensus_labels = np.array(consensus_labels)
        
        anomaly_count = np.sum(consensus_labels == 0)
        normal_count = np.sum(consensus_labels == 1)
        logger.info(f"Consensus: {anomaly_count} anomalies, {normal_count} normal")
        
        return consensus_labels
    
    def validate_labels(self, X: np.ndarray, labels: np.ndarray, method_name: str) -> Dict[str, float]:
        """Validate synthetic labels using statistical properties"""
        normal_mask = labels == 1
        anomaly_mask = labels == 0
        
        normal_data = X[normal_mask]
        anomaly_data = X[anomaly_mask]
        
        if len(anomaly_data) == 0:
            return {"error": "No anomalies detected"}
        
        validation_metrics = {}
        
        # Temperature analysis (assuming temperature is in column 2)
        if X.shape[1] > 2:
            normal_temps = normal_data[:, 2]
            anomaly_temps = anomaly_data[:, 2]
            
            temp_extremes = np.abs(anomaly_temps - np.mean(normal_temps)) / np.std(normal_temps)
            validation_metrics['temp_extremes'] = float(np.mean(temp_extremes))
        
        # Voltage analysis (assuming voltage is in column 0)
        if X.shape[1] > 0:
            normal_volts = normal_data[:, 0]
            anomaly_volts = anomaly_data[:, 0]
            
            volt_std_normal = np.std(normal_volts)
            volt_std_anomaly = np.std(anomaly_volts)
            validation_metrics['voltage_fluctuation_ratio'] = float(volt_std_anomaly / volt_std_normal)
        
        # SoC analysis (assuming SoC is in column 3)
        if X.shape[1] > 3:
            normal_soc = normal_data[:, 3]
            anomaly_soc = anomaly_data[:, 3]
            
            soc_variance_normal = np.var(normal_soc)
            soc_variance_anomaly = np.var(anomaly_soc)
            validation_metrics['soc_variance_ratio'] = float(soc_variance_anomaly / soc_variance_normal)
        
        return validation_metrics

class SyntheticLabelPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.labels_dir = self.processed_dir / "labels"
        self.features_dir = self.processed_dir / "features"
        
        # Create label subdirectories
        for method in ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'elliptic_envelope', 'consensus']:
            (self.labels_dir / method).mkdir(parents=True, exist_ok=True)
        
        self.label_generator = SyntheticLabelGenerator()
    
    def load_features(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load extracted features"""
        features_file = self.features_dir / "extracted_features.parquet"
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features not found at {features_file}. Run preprocessing first.")
        
        features_df = pd.read_parquet(features_file)
        
        # Extract feature columns (exclude metadata)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['battery_id', 'battery_type', 'time']]
        
        X = features_df[feature_columns].values
        logger.info(f"Loaded features: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, features_df
    
    def generate_synthetic_labels(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate synthetic labels using all methods"""
        logger.info("Generating synthetic labels using multiple methods...")
        
        # Generate labels using all methods
        all_labels = self.label_generator.generate_all_labels(X)
        
        # Generate consensus labels
        consensus_labels = self.label_generator.generate_consensus_labels(all_labels)
        all_labels['consensus'] = consensus_labels
        
        return all_labels
    
    def validate_all_labels(self, X: np.ndarray, all_labels: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Validate all generated labels"""
        logger.info("Validating synthetic labels...")
        
        validation_results = {}
        
        for method_name, labels in all_labels.items():
            if labels is not None:
                logger.info(f"Validating {method_name} labels...")
                validation_metrics = self.label_generator.validate_labels(X, labels, method_name)
                validation_results[method_name] = validation_metrics
                
                # Log validation results
                if 'error' not in validation_metrics:
                    logger.info(f"  {method_name} validation:")
                    for metric, value in validation_metrics.items():
                        logger.info(f"    {metric}: {value:.3f}")
                else:
                    logger.warning(f"  {method_name}: {validation_metrics['error']}")
        
        return validation_results
    
    def save_labels(self, all_labels: Dict[str, np.ndarray], validation_results: Dict[str, Dict[str, float]]):
        """Save all generated labels and validation results"""
        logger.info("Saving synthetic labels...")
        
        # Save individual method labels
        for method_name, labels in all_labels.items():
            if labels is not None:
                method_dir = self.labels_dir / method_name
                labels_file = method_dir / "labels.npy"
                np.save(labels_file, labels)
                logger.info(f"Saved {method_name} labels to {labels_file}")
        
        # Save validation results
        validation_file = self.labels_dir / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Saved validation results to {validation_file}")
        
        # Save consensus labels as the main labels
        if 'consensus' in all_labels and all_labels['consensus'] is not None:
            consensus_file = self.labels_dir / "consensus_labels.npy"
            np.save(consensus_file, all_labels['consensus'])
            logger.info(f"Saved consensus labels to {consensus_file}")
    
    def run_synthetic_label_generation(self):
        """Run complete synthetic label generation pipeline"""
        logger.info("Starting synthetic label generation pipeline...")
        
        # Step 1: Load features
        logger.info("Step 1: Loading features...")
        X, features_df = self.load_features()
        
        # Step 2: Generate synthetic labels
        logger.info("Step 2: Generating synthetic labels...")
        all_labels = self.generate_synthetic_labels(X)
        
        # Step 3: Validate labels
        logger.info("Step 3: Validating labels...")
        validation_results = self.validate_all_labels(X, all_labels)
        
        # Step 4: Save labels
        logger.info("Step 4: Saving labels...")
        self.save_labels(all_labels, validation_results)
        
        # Summary
        logger.info("="*50)
        logger.info("SYNTHETIC LABEL GENERATION SUMMARY")
        logger.info("="*50)
        
        for method_name, labels in all_labels.items():
            if labels is not None:
                anomaly_count = np.sum(labels == 0)
                normal_count = np.sum(labels == 1)
                anomaly_rate = anomaly_count / len(labels)
                logger.info(f"{method_name}: {anomaly_count} anomalies ({anomaly_rate:.1%}), {normal_count} normal")
        
        # Validation summary
        logger.info("\nValidation Results:")
        for method_name, metrics in validation_results.items():
            if 'error' not in metrics:
                logger.info(f"{method_name}:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.3f}")
        
        logger.info("Synthetic label generation completed successfully!")
        return all_labels, validation_results

def main():
    """Main function to run synthetic label generation"""
    pipeline = SyntheticLabelPipeline()
    all_labels, validation_results = pipeline.run_synthetic_label_generation()
    
    if all_labels is not None:
        print(f"\n✅ Synthetic label generation completed!")
        print(f"📊 Generated labels for {len(all_labels)} methods")
        print(f"📁 Labels saved to: data/processed/labels/")
        print(f"📁 Validation results saved to: data/processed/labels/validation_results.json")
    else:
        print("❌ Synthetic label generation failed!")

if __name__ == "__main__":
    main()
