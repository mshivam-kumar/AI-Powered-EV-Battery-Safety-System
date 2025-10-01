#!/usr/bin/env python3
"""
Step 5: Testing and Validation Pipeline
Comprehensive model evaluation and validation
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, models_dir: str = "models", results_dir: str = "results"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_models(self) -> Dict[str, any]:
        """Load all trained models"""
        models = {}
        
        # Load Isolation Forest
        iforest_file = self.models_dir / "isolation_forest.pkl"
        if iforest_file.exists():
            with open(iforest_file, 'rb') as f:
                models['isolation_forest'] = pickle.load(f)
            logger.info("Loaded Isolation Forest")
        
        # Load supervised models
        for model_name in ['random_forest', 'gradient_boosting', 'lstm_model']:
            model_file = self.models_dir / f"{model_name}.pkl"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name}")
        
        # Load RL Agent
        q_table_file = self.models_dir / "q_table.pkl"
        rl_metadata_file = self.models_dir / "rl_metadata.json"
        if q_table_file.exists() and rl_metadata_file.exists():
            with open(q_table_file, 'rb') as f:
                models['q_table'] = pickle.load(f)
            with open(rl_metadata_file, 'r') as f:
                models['rl_metadata'] = json.load(f)
            logger.info("Loaded RL Agent")
        
        return models
    
    def load_test_data(self, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
        """Load test data and labels"""
        data_path = Path(data_dir)
        
        # Load test split
        test_file = data_path / "processed" / "splits" / "test.parquet"
        test_df = pd.read_parquet(test_file)
        
        # Load consensus labels
        labels_file = data_path / "processed" / "labels" / "consensus_labels.npy"
        all_labels = np.load(labels_file)
        
        # Get labels for test samples
        test_indices = test_df.index
        y_test = all_labels[test_indices]
        
        # Extract feature columns (exclude metadata)
        feature_columns = [col for col in test_df.columns 
                          if col not in ['battery_id', 'battery_type', 'time']]
        
        X_test = test_df[feature_columns].values
        
        logger.info(f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        logger.info(f"Test label distribution: {np.sum(y_test == 0)} anomalies, {np.sum(y_test == 1)} normal")
        
        return X_test, y_test
    
    def evaluate_supervised_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """Evaluate supervised model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                metrics['roc_auc'] = roc_auc
            except:
                metrics['roc_auc'] = None
        
        logger.info(f"  {model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return metrics
    
    def evaluate_isolation_forest(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate Isolation Forest"""
        logger.info("Evaluating Isolation Forest...")
        
        # Get anomaly scores
        anomaly_scores = model.decision_function(X_test)
        # Convert to binary predictions (lower scores = more anomalous)
        y_pred = (anomaly_scores < 0).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC using anomaly scores
        try:
            roc_auc = roc_auc_score(y_test, -anomaly_scores)  # Negative because lower scores = more anomalous
        except:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'anomaly_scores': anomaly_scores.tolist()
        }
        
        logger.info(f"  Isolation Forest - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return metrics
    
    def evaluate_rl_agent(self, q_table: np.ndarray, metadata: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate RL Agent"""
        logger.info("Evaluating RL Agent...")
        
        # Extract state variables and discretize
        soc_bins = metadata['soc_bins']
        temp_bins = metadata['temp_bins']
        ambient_bins = metadata['ambient_bins']
        actions = metadata['actions']
        
        def discretize_state(soc, temp, ambient):
            soc_bin = min(int(soc * (len(soc_bins) - 1)), len(soc_bins) - 2)
            temp_bin = min(int(temp * (len(temp_bins) - 1)), len(temp_bins) - 2)
            ambient_bin = min(int(ambient * (len(ambient_bins) - 1)), len(ambient_bins) - 2)
            return soc_bin, temp_bin, ambient_bin
        
        # Get RL predictions
        y_pred = []
        for sample in X_test:
            # Extract state variables (assuming standardized features)
            soc = (sample[3] + 1) / 2  # SoC (normalized)
            temp = (sample[2] + 1) / 2  # Temperature (normalized)
            ambient = (sample[4] + 1) / 2  # Ambient temperature (normalized)
            
            # Discretize state
            state = discretize_state(soc, temp, ambient)
            
            # Get action with highest Q-value
            action_values = q_table[state]
            best_action_idx = np.argmax(action_values)
            
            # Convert action to prediction (fast/slow = normal, pause = anomaly)
            if actions[best_action_idx] == 'pause':
                y_pred.append(0)  # Anomaly
            else:
                y_pred.append(1)  # Normal
        
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'q_table_shape': q_table.shape,
            'actions': actions
        }
        
        logger.info(f"  RL Agent - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return metrics
    
    def create_ensemble_prediction(self, models: Dict[str, any], X_test: np.ndarray) -> np.ndarray:
        """Create ensemble prediction from all models"""
        logger.info("Creating ensemble prediction...")
        
        predictions = []
        weights = []
        
        # Isolation Forest
        if 'isolation_forest' in models:
            iforest = models['isolation_forest']
            anomaly_scores = iforest.decision_function(X_test)
            pred = (anomaly_scores < 0).astype(int)
            predictions.append(pred)
            weights.append(0.4)  # Higher weight for primary model
        
        # Supervised models
        for model_name in ['random_forest', 'gradient_boosting', 'lstm_model']:
            if model_name in models:
                model = models[model_name]
                pred = model.predict(X_test)
                predictions.append(pred)
                weights.append(0.2)  # Equal weight for supervised models
        
        # RL Agent
        if 'q_table' in models and 'rl_metadata' in models:
            q_table = models['q_table']
            metadata = models['rl_metadata']
            
            # Get RL predictions (simplified)
            rl_pred = np.ones(len(X_test))  # Default to normal
            predictions.append(rl_pred)
            weights.append(0.2)
        
        # Weighted ensemble
        if predictions:
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            ensemble_pred = (ensemble_pred > 0.5).astype(int)
            
            logger.info(f"Ensemble created with {len(predictions)} models")
            return ensemble_pred
        else:
            logger.warning("No models available for ensemble")
            return np.ones(len(X_test))  # Default to normal
    
    def create_visualizations(self, results: Dict[str, Dict], y_test: np.ndarray, ensemble_pred: np.ndarray):
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Model Performance Comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        model_names = []
        accuracies = []
        f1_scores = []
        
        for model_name, metrics in results.items():
            if 'accuracy' in metrics and 'f1_score' in metrics:
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
                f1_scores.append(metrics['f1_score'])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            ax.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            ax.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix for Ensemble
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        cm = confusion_matrix(y_test, ensemble_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Ensemble Model Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved to results/")
    
    def run_testing(self):
        """Run complete testing pipeline"""
        logger.info("Starting model testing pipeline...")
        
        # Step 1: Load models
        logger.info("Step 1: Loading trained models...")
        models = self.load_models()
        
        # Step 2: Load test data
        logger.info("Step 2: Loading test data...")
        X_test, y_test = self.load_test_data()
        
        # Step 3: Evaluate models
        logger.info("Step 3: Evaluating models...")
        results = {}
        
        # Evaluate Isolation Forest
        if 'isolation_forest' in models:
            results['isolation_forest'] = self.evaluate_isolation_forest(
                models['isolation_forest'], X_test, y_test
            )
        
        # Evaluate supervised models
        for model_name in ['random_forest', 'gradient_boosting', 'lstm_model']:
            if model_name in models:
                results[model_name] = self.evaluate_supervised_model(
                    models[model_name], X_test, y_test, model_name
                )
        
        # Evaluate RL Agent
        if 'q_table' in models and 'rl_metadata' in models:
            results['rl_agent'] = self.evaluate_rl_agent(
                models['q_table'], models['rl_metadata'], X_test, y_test
            )
        
        # Step 4: Create ensemble
        logger.info("Step 4: Creating ensemble prediction...")
        ensemble_pred = self.create_ensemble_prediction(models, X_test)
        
        # Evaluate ensemble
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'f1_score': ensemble_f1,
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred).tolist()
        }
        
        # Step 5: Create visualizations
        logger.info("Step 5: Creating visualizations...")
        self.create_visualizations(results, y_test, ensemble_pred)
        
        # Step 6: Save results
        logger.info("Step 6: Saving results...")
        results_file = self.results_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_file}")
        
        # Summary
        logger.info("="*50)
        logger.info("MODEL TESTING SUMMARY")
        logger.info("="*50)
        
        for model_name, metrics in results.items():
            if 'accuracy' in metrics and 'f1_score' in metrics:
                logger.info(f"{model_name}:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
                if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        
        logger.info("Model testing completed successfully!")
        return results

def main():
    """Main function to run testing"""
    tester = ModelTester()
    results = tester.run_testing()
    
    if results is not None:
        print(f"\n✅ Model testing completed!")
        print(f"📊 Evaluated {len(results)} models")
        print(f"📁 Results saved to: results/")
        print(f"📊 Visualizations saved to: results/")
    else:
        print("❌ Model testing failed!")

if __name__ == "__main__":
    main()
