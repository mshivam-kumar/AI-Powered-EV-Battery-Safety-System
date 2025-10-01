#!/usr/bin/env python3
"""
Step 4: Model Training Pipeline
Train all models: Isolation Forest, Random Forest, Gradient Boosting, LSTM (MLP), RL Agent
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'isolation_forest': {
                'class': IsolationForest,
                'params': {
                    'n_estimators': 100,
                    'contamination': 0.02,
                    'random_state': 42
                }
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'lstm_model': {
                'class': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (64, 32),
                    'max_iter': 200,
                    'random_state': 42,
                    'early_stopping': True,
                    'validation_fraction': 0.1
                }
            }
        }
    
    def load_training_data(self, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
        """Load training data and labels"""
        data_path = Path(data_dir)
        
        # Load features
        features_file = data_path / "processed" / "features" / "extracted_features.parquet"
        features_df = pd.read_parquet(features_file)
        
        # Load consensus labels
        labels_file = data_path / "processed" / "labels" / "consensus_labels.npy"
        labels = np.load(labels_file)
        
        # Extract feature columns (exclude metadata)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['battery_id', 'battery_type', 'time']]
        
        X = features_df[feature_columns].values
        
        logger.info(f"Loaded training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Label distribution: {np.sum(labels == 0)} anomalies, {np.sum(labels == 1)} normal")
        
        return X, labels
    
    def train_isolation_forest(self, X: np.ndarray) -> IsolationForest:
        """Train Isolation Forest (unsupervised)"""
        logger.info("Training Isolation Forest...")
        
        config = self.model_configs['isolation_forest']
        model = config['class'](**config['params'])
        
        model.fit(X)
        
        # Save model
        model_file = self.models_dir / "isolation_forest.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved Isolation Forest to {model_file}")
        
        return model
    
    def train_supervised_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
        """Train supervised models"""
        trained_models = {}
        
        for model_name in ['random_forest', 'gradient_boosting', 'lstm_model']:
            logger.info(f"Training {model_name}...")
            
            config = self.model_configs[model_name]
            model = config['class'](**config['params'])
            
            # Train model
            model.fit(X, y)
            
            # Evaluate model
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            logger.info(f"  {model_name} training accuracy: {accuracy:.3f}")
            logger.info(f"  {model_name} precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
            
            # Save model
            model_file = self.models_dir / f"{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  Saved {model_name} to {model_file}")
            
            trained_models[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return trained_models
    
    def train_rl_agent(self, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
        """Train RL Agent (Q-Learning)"""
        logger.info("Training RL Agent (Q-Learning)...")
        
        # RL parameters
        ALPHA = 0.2  # Learning rate
        GAMMA = 0.95  # Discount factor
        EPSILON = 0.05  # Exploration rate
        
        # State space (discretized)
        SOC_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        TEMP_BINS = [0.0, 20.0, 30.0, 40.0, 50.0, 100.0]
        AMBIENT_BINS = [0.0, 20.0, 30.0, 40.0, 60.0, 100.0]
        
        # Action space
        ACTIONS = ['fast', 'slow', 'pause']
        
        # Initialize Q-table
        n_soc_bins = len(SOC_BINS) - 1
        n_temp_bins = len(TEMP_BINS) - 1
        n_ambient_bins = len(AMBIENT_BINS) - 1
        n_actions = len(ACTIONS)
        
        Q_table = np.zeros((n_soc_bins, n_temp_bins, n_ambient_bins, n_actions))
        
        def discretize_state(soc, temp, ambient):
            """Discretize continuous state to discrete bins"""
            soc_bin = min(int(soc * n_soc_bins), n_soc_bins - 1)
            temp_bin = min(int(temp * n_temp_bins), n_temp_bins - 1)
            ambient_bin = min(int(ambient * n_ambient_bins), n_ambient_bins - 1)
            return soc_bin, temp_bin, ambient_bin
        
        def compute_reward(soc, temp, ambient, is_anomaly):
            """Compute reward for RL agent"""
            reward = 0
            
            # Base reward for normal operation
            if not is_anomaly:
                reward += 1.0
            
            # Penalty for anomalies
            if is_anomaly:
                reward -= 2.0
            
            # Reward for optimal SoC range
            if 0.3 <= soc <= 0.8:
                reward += 0.5
            
            # Penalty for extreme temperatures
            if temp > 0.8:  # High temperature
                reward -= 1.0
            
            # Penalty for extreme ambient
            if ambient > 0.8:  # High ambient temperature
                reward -= 0.5
            
            return reward
        
        # Training loop
        n_episodes = 1000
        episode_rewards = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            
            # Sample random subset of data for this episode
            n_samples = min(100, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            
            for i in indices:
                sample = X[i]
                is_anomaly = y[i] == 0
                
                # Extract state variables (assuming standardized features)
                soc = (sample[3] + 1) / 2  # SoC (normalized)
                temp = (sample[2] + 1) / 2  # Temperature (normalized)
                ambient = (sample[4] + 1) / 2  # Ambient temperature (normalized)
                
                # Discretize state
                state = discretize_state(soc, temp, ambient)
                
                # Choose action (epsilon-greedy)
                if np.random.random() < EPSILON:
                    action = np.random.choice(n_actions)
                else:
                    action = np.argmax(Q_table[state])
                
                # Compute reward
                reward = compute_reward(soc, temp, ambient, is_anomaly)
                episode_reward += reward
                
                # Update Q-table
                old_value = Q_table[state][action]
                max_future_q = np.max(Q_table[state])
                new_value = old_value + ALPHA * (reward + GAMMA * max_future_q - old_value)
                Q_table[state][action] = new_value
            
            episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"  Episode {episode}, Average Reward: {avg_reward:.3f}")
        
        # Save Q-table
        q_table_file = self.models_dir / "q_table.pkl"
        with open(q_table_file, 'wb') as f:
            pickle.dump(Q_table, f)
        logger.info(f"Saved Q-table to {q_table_file}")
        
        # Save RL metadata
        rl_metadata = {
            'alpha': ALPHA,
            'gamma': GAMMA,
            'epsilon': EPSILON,
            'n_episodes': n_episodes,
            'state_space': (n_soc_bins, n_temp_bins, n_ambient_bins),
            'action_space': n_actions,
            'actions': ACTIONS,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'soc_bins': SOC_BINS,
            'temp_bins': TEMP_BINS,
            'ambient_bins': AMBIENT_BINS
        }
        
        metadata_file = self.models_dir / "rl_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(rl_metadata, f, indent=2)
        logger.info(f"Saved RL metadata to {metadata_file}")
        
        return {
            'q_table': Q_table,
            'metadata': rl_metadata,
            'episode_rewards': episode_rewards
        }
    
    def run_training(self):
        """Run complete training pipeline"""
        logger.info("Starting model training pipeline...")
        
        # Step 1: Load training data
        logger.info("Step 1: Loading training data...")
        X, y = self.load_training_data()
        
        # Step 2: Train Isolation Forest
        logger.info("Step 2: Training Isolation Forest...")
        iforest = self.train_isolation_forest(X)
        
        # Step 3: Train supervised models
        logger.info("Step 3: Training supervised models...")
        supervised_models = self.train_supervised_models(X, y)
        
        # Step 4: Train RL Agent
        logger.info("Step 4: Training RL Agent...")
        rl_agent = self.train_rl_agent(X, y)
        
        # Summary
        logger.info("="*50)
        logger.info("MODEL TRAINING SUMMARY")
        logger.info("="*50)
        
        logger.info("Isolation Forest: Trained (unsupervised)")
        
        for model_name, metrics in supervised_models.items():
            logger.info(f"{model_name}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall: {metrics['recall']:.3f}")
            logger.info(f"  F1-Score: {metrics['f1']:.3f}")
        
        logger.info(f"RL Agent:")
        logger.info(f"  Final Average Reward: {rl_agent['metadata']['final_avg_reward']:.3f}")
        logger.info(f"  Q-table Shape: {rl_agent['q_table'].shape}")
        logger.info(f"  Actions: {rl_agent['metadata']['actions']}")
        
        logger.info("Model training completed successfully!")
        
        return {
            'isolation_forest': iforest,
            'supervised_models': supervised_models,
            'rl_agent': rl_agent
        }

def main():
    """Main function to run training"""
    trainer = ModelTrainer()
    results = trainer.run_training()
    
    if results is not None:
        print(f"\n✅ Model training completed!")
        print(f"📊 Trained 5 models: IF, RF, GB, LSTM, RL")
        print(f"📁 Models saved to: models/")
    else:
        print("❌ Model training failed!")

if __name__ == "__main__":
    main()
