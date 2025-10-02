#!/usr/bin/env python3
"""
Robust RL Agent Training - Leveraging Feature Analysis Insights
Train an advanced RL agent using insights from comprehensive feature analysis
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import time
from datetime import datetime
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustRLTrainer:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load feature analysis insights
        self.load_feature_insights()
        
        # Enhanced RL parameters based on feature analysis
        self.ACTIONS = ['fast_charge', 'slow_charge', 'maintain', 'discharge', 'pause']
        self.ACTION_INDICES = {action: i for i, action in enumerate(self.ACTIONS)}
        
        # Improved hyperparameters
        self.ALPHA = 0.1      # Learning rate
        self.GAMMA = 0.95     # Discount factor (higher for long-term safety)
        self.EPSILON = 0.3    # Exploration rate (higher for better exploration)
        self.EPSILON_DECAY = 0.995
        self.MIN_EPSILON = 0.01
        
        # Initialize epsilon for training
        self.epsilon = self.EPSILON
        
        # Enhanced state discretization based on feature analysis
        self.setup_enhanced_state_space()
        
    def load_feature_insights(self):
        """Load insights from comprehensive feature analysis"""
        feature_analysis_path = self.models_dir / "comprehensive_feature_analysis.json"
        
        if feature_analysis_path.exists():
            with open(feature_analysis_path, 'r') as f:
                self.feature_analysis = json.load(f)
            print("✅ Loaded feature analysis insights")
            
            # Extract critical thresholds for reward function
            self.critical_thresholds = {}
            for feature_data in self.feature_analysis:
                feature_name = feature_data['feature_name']
                if feature_data.get('best_anomaly_threshold'):
                    self.critical_thresholds[feature_name] = {
                        'threshold_value': feature_data['best_anomaly_threshold']['value'],
                        'anomaly_rate': feature_data['best_anomaly_threshold']['anomaly_rate'],
                        'threshold_type': feature_data['best_anomaly_threshold']['name']
                    }
            
            print(f"📊 Loaded {len(self.critical_thresholds)} critical thresholds")
        else:
            print("⚠️ Feature analysis not found, using default thresholds")
            self.feature_analysis = []
            self.critical_thresholds = {}
    
    def setup_enhanced_state_space(self):
        """Setup enhanced state space based on feature importance"""
        # Based on our analysis, focus on top predictive features:
        # 1. C-Rate (12.3% importance)
        # 2. Power (10.9% importance) 
        # 3. Current (10.7% importance)
        # 4. Voltage Gradient (9.0% importance)
        # 5. Temperature (6.6% importance)
        # 6. SoC (0.5% importance but 99.8% anomaly rate at extremes)
        
        # Enhanced discretization with more granular bins for critical features
        self.C_RATE_BINS = np.array([-1.0, 0.5, 1.5, 2.5, 4.0, 6.0])  # Based on analysis thresholds
        self.POWER_BINS = np.array([-2.0, 0.0, 1.5, 2.5, 4.0, 6.0])   # Based on analysis thresholds
        self.TEMP_BINS = np.array([-2.0, 0.0, 1.5, 2.5, 4.0, 6.0])    # Based on analysis thresholds
        self.SOC_BINS = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 3.0])   # Based on analysis thresholds
        self.VOLTAGE_BINS = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 3.0]) # Based on analysis thresholds
        
        # Calculate state space size
        self.state_space_size = (
            len(self.C_RATE_BINS) - 1,
            len(self.POWER_BINS) - 1, 
            len(self.TEMP_BINS) - 1,
            len(self.SOC_BINS) - 1,
            len(self.VOLTAGE_BINS) - 1,
            2  # Anomaly flag (0 or 1)
        )
        
        total_states = np.prod(self.state_space_size)
        print(f"🧠 Enhanced state space: {self.state_space_size} = {total_states:,} total states")
        
        # Initialize Q-table
        self.Q_table = np.zeros(self.state_space_size + (len(self.ACTIONS),))
        
    def discretize_enhanced_state(self, c_rate, power, temp, soc, voltage, is_anomaly):
        """Enhanced state discretization using feature analysis insights"""
        c_rate_bin = np.digitize(c_rate, self.C_RATE_BINS) - 1
        power_bin = np.digitize(power, self.POWER_BINS) - 1
        temp_bin = np.digitize(temp, self.TEMP_BINS) - 1
        soc_bin = np.digitize(soc, self.SOC_BINS) - 1
        voltage_bin = np.digitize(voltage, self.VOLTAGE_BINS) - 1
        anomaly_bin = int(is_anomaly)
        
        # Ensure bins are within valid range
        c_rate_bin = np.clip(c_rate_bin, 0, len(self.C_RATE_BINS) - 2)
        power_bin = np.clip(power_bin, 0, len(self.POWER_BINS) - 2)
        temp_bin = np.clip(temp_bin, 0, len(self.TEMP_BINS) - 2)
        soc_bin = np.clip(soc_bin, 0, len(self.SOC_BINS) - 2)
        voltage_bin = np.clip(voltage_bin, 0, len(self.VOLTAGE_BINS) - 2)
        
        return (c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin)
    
    def compute_enhanced_reward(self, c_rate, power, temp, soc, voltage, is_anomaly, action_idx):
        """Enhanced reward function based on feature analysis insights"""
        action = self.ACTIONS[action_idx]
        
        # Start with base reward
        reward = 0.0
        
        # CRITICAL SAFETY VIOLATIONS - Based on feature analysis
        # These are the conditions with highest anomaly rates from our analysis
        
        # 1. SoC Critical Depletion (99.8% anomaly rate at <1st percentile)
        if soc < -3.5:  # Critically low SoC
            if action in ['discharge', 'maintain']:
                return -10000.0  # MASSIVE penalty for dangerous actions
            elif action == 'pause':
                return -5000.0   # Still bad - battery needs charging
            elif action in ['slow_charge', 'fast_charge']:
                reward += 1000.0  # HUGE reward for necessary charging
        
        # 2. Power Stress (79.5% anomaly rate at >99th percentile)
        if power > 3.5:  # Extreme power conditions
            if action == 'fast_charge':
                return -8000.0   # Never fast charge during power stress
            elif action == 'pause':
                reward += 800.0  # Excellent - pause during power stress
            elif action == 'discharge':
                reward += 600.0  # Good - reduce power load
        
        # 3. Temperature Gradient Critical (78.0% anomaly rate at >99th percentile)
        # Note: We don't have temp_gradient in state, but use temperature as proxy
        if temp > 3.5:  # Extreme temperature (equivalent to >99th percentile)
            if action in ['fast_charge', 'slow_charge']:
                return -7000.0   # Never charge when extremely hot
            elif action == 'pause':
                reward += 700.0  # Excellent - pause when overheating
            elif action == 'discharge':
                reward += 500.0  # Good - discharge to reduce heat
        
        # 4. Voltage Critical Issues (66.3% anomaly rate at <1st percentile)
        if voltage < -3.5:  # Critically low voltage
            if action in ['discharge', 'maintain']:
                return -6000.0   # Don't discharge when voltage is critical
            elif action in ['slow_charge', 'fast_charge']:
                reward += 600.0  # Need to restore voltage
        
        # 5. C-Rate Extreme (64.0% anomaly rate at >99th percentile)
        if c_rate > 4.0:  # Extreme charging rate
            if action == 'fast_charge':
                return -5000.0   # Don't add to extreme charging
            elif action == 'pause':
                reward += 500.0  # Good - pause extreme charging
            elif action == 'slow_charge':
                reward += 300.0  # Better - reduce charging rate
        
        # ANOMALY DETECTION INTEGRATION
        if is_anomaly:
            # If Random Forest detects anomaly, be extra cautious
            if action in ['fast_charge', 'slow_charge']:
                reward -= 3000.0  # Heavy penalty for charging during detected anomaly
            elif action == 'pause':
                reward += 400.0   # Good - pause during anomaly
            elif action == 'discharge':
                reward += 200.0   # Okay - discharge during anomaly
        
        # NORMAL OPERATION REWARDS (when not in critical states)
        if not is_anomaly and soc > -3.5 and temp < 2.0 and power < 2.0 and voltage > -2.0:
            # Optimal SoC range (-2.0 to 0.0 based on analysis)
            if -2.0 <= soc <= 0.0:
                reward += 50.0  # Bonus for optimal SoC range
                
                # Temperature-based charging decisions
                if temp < 0.0:  # Cool conditions - safe for fast charging
                    if action == 'fast_charge':
                        reward += 100.0
                    elif action == 'slow_charge':
                        reward += 80.0
                elif temp < 1.5:  # Moderate temperature - prefer slow charging
                    if action == 'slow_charge':
                        reward += 60.0
                    elif action == 'fast_charge':
                        reward += 20.0
                else:  # Getting warm - be cautious
                    if action == 'maintain':
                        reward += 40.0
                    elif action == 'slow_charge':
                        reward += 20.0
            
            # SoC too low (but not critical)
            elif soc < -2.0:
                if action in ['slow_charge', 'fast_charge']:
                    reward += 30.0  # Need to charge
                elif action in ['discharge', 'maintain']:
                    reward -= 100.0  # Don't discharge when low
            
            # SoC too high
            elif soc > 0.0:
                if action == 'discharge':
                    reward += 40.0  # Good - discharge when high
                elif action == 'maintain':
                    reward += 20.0  # Okay - maintain when high
                elif action in ['fast_charge', 'slow_charge']:
                    reward -= 200.0  # Don't charge when already high
        
        # EFFICIENCY BONUS - Reward actions that align with feature importance
        # C-Rate is most important feature, so reward appropriate C-rate management
        if -1.0 <= c_rate <= 1.5:  # Normal C-rate range
            reward += 10.0
        
        return reward
    
    def load_training_data(self, data_dir: str = "data"):
        """Load training data with enhanced features"""
        print("📂 Loading training data for RL...")
        
        splits_dir = Path(data_dir) / "processed" / "splits_complete"
        
        # Load training features and labels
        X_train = pd.read_parquet(splits_dir / "train_features.parquet")
        y_train = pd.read_csv(splits_dir / "train_labels.csv")['anomaly'].values
        
        # Remove metadata columns
        feature_cols = [col for col in X_train.columns if col not in ['battery_id', 'battery_type', 'time']]
        X_train = X_train[feature_cols]
        
        print(f"✅ Loaded {len(X_train):,} training samples with {len(feature_cols)} features")
        
        # Extract key features for RL state representation
        rl_features = {
            'c_rate': X_train['c_rate'].values if 'c_rate' in X_train.columns else np.zeros(len(X_train)),
            'power': X_train['power'].values if 'power' in X_train.columns else np.zeros(len(X_train)),
            'temperature': X_train['temperature'].values if 'temperature' in X_train.columns else np.zeros(len(X_train)),
            'soc': X_train['soc'].values if 'soc' in X_train.columns else np.zeros(len(X_train)),
            'voltage': X_train['voltage'].values if 'voltage' in X_train.columns else np.zeros(len(X_train)),
            'anomaly': y_train
        }
        
        return rl_features
    
    def train_robust_agent(self, training_data, episodes=15000, episode_samples=1000):
        """Train robust RL agent using enhanced reward function"""
        print(f"\n🤖 Training Robust RL Agent")
        print("=" * 60)
        print(f"Episodes: {episodes:,}")
        print(f"Samples per episode: {episode_samples:,}")
        print(f"Total training interactions: {episodes * episode_samples:,}")
        
        # Training metrics
        episode_rewards = []
        safety_scores = []
        exploration_rates = []
        q_value_evolution = []
        action_history = []
        
        start_time = time.time()
        
        print(f"🎯 Starting training with enhanced logging...")
        print(f"📊 Initial Q-table size: {self.Q_table.size:,} total states")
        
        for episode in range(episodes):
            episode_reward = 0.0
            episode_start = time.time()
            episode_actions = []
            
            # Sample random states from training data
            sample_indices = np.random.choice(len(training_data['c_rate']), episode_samples, replace=True)
            
            for i, idx in enumerate(sample_indices):
                # Current state
                c_rate = training_data['c_rate'][idx]
                power = training_data['power'][idx]
                temp = training_data['temperature'][idx]
                soc = training_data['soc'][idx]
                voltage = training_data['voltage'][idx]
                is_anomaly = training_data['anomaly'][idx]
                
                state = self.discretize_enhanced_state(c_rate, power, temp, soc, voltage, is_anomaly)
                
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action_idx = np.random.choice(len(self.ACTIONS))  # Explore
                else:
                    action_idx = np.argmax(self.Q_table[state])  # Exploit
                
                episode_actions.append(action_idx)
                
                # Compute reward
                reward = self.compute_enhanced_reward(c_rate, power, temp, soc, voltage, is_anomaly, action_idx)
                episode_reward += reward
                
                # Simulate next state (simplified - assume same state for now)
                next_state = state
                
                # Q-learning update
                current_q = self.Q_table[state + (action_idx,)]
                max_next_q = np.max(self.Q_table[next_state])
                
                new_q = current_q + self.ALPHA * (reward + self.GAMMA * max_next_q - current_q)
                self.Q_table[state + (action_idx,)] = new_q
            
            # Episode statistics
            episode_rewards.append(episode_reward / episode_samples)
            action_history.extend(episode_actions)
            
            # Keep only recent actions for distribution analysis
            if len(action_history) > 10000:
                action_history = action_history[-10000:]
            
            # Decay exploration rate
            self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.EPSILON_DECAY)
            exploration_rates.append(self.epsilon)
            
            # Comprehensive logging every 500 episodes
            if episode % 500 == 0:
                print(f"\n{'='*80}")
                print(f"🔄 EPISODE {episode:,} PROGRESS REPORT")
                print(f"{'='*80}")
                
                # Safety validation with detailed logging for key episodes
                detailed_logging = (episode % 2000 == 0)  # Detailed every 2000 episodes
                if detailed_logging:
                    result = self.quick_safety_validation(episode, detailed_logging=True)
                    if isinstance(result, tuple):
                        safety_score, detailed_results = result
                    else:
                        safety_score = result
                        detailed_results = None
                else:
                    result = self.quick_safety_validation(episode, detailed_logging=False)
                    if isinstance(result, tuple):
                        safety_score = result[0]
                    else:
                        safety_score = result
                
                safety_scores.append(safety_score)
                
                # Q-table analysis
                q_stats = self.analyze_q_table_evolution(episode)
                if q_stats:
                    q_value_evolution.append(q_stats['mean'])
                
                # Action distribution analysis
                recent_actions = action_history[-1000:] if len(action_history) >= 1000 else action_history
                action_dist = self.log_action_distribution(episode, recent_actions)
                
                # Performance metrics
                episode_time = time.time() - episode_start
                elapsed_time = time.time() - start_time
                avg_reward = episode_rewards[-1]
                
                print(f"⏱️  Training Progress:")
                print(f"   Episode: {episode:,} / {episodes:,} ({episode/episodes*100:.1f}%)")
                print(f"   Elapsed time: {elapsed_time:.0f}s ({elapsed_time/60:.1f} min)")
                print(f"   Episode time: {episode_time:.3f}s")
                print(f"   ETA: {(elapsed_time/max(episode,1))*(episodes-episode):.0f}s")
                
                print(f"🎯 Learning Metrics:")
                print(f"   Average reward: {avg_reward:8.1f}")
                print(f"   Safety score: {safety_score:5.1f}%")
                print(f"   Exploration rate: {self.epsilon:.3f}")
                
                # Learning progress indicators
                if len(safety_scores) > 1:
                    safety_trend = safety_scores[-1] - safety_scores[-2]
                    trend_arrow = "📈" if safety_trend > 0 else "📉" if safety_trend < 0 else "➡️"
                    print(f"   Safety trend: {trend_arrow} {safety_trend:+.1f}%")
                
                if len(episode_rewards) >= 100:
                    recent_reward_trend = np.mean(episode_rewards[-50:]) - np.mean(episode_rewards[-100:-50])
                    reward_arrow = "📈" if recent_reward_trend > 0 else "📉" if recent_reward_trend < 0 else "➡️"
                    print(f"   Reward trend: {reward_arrow} {recent_reward_trend:+.1f}")
                
                print(f"{'='*80}")
            
            # Quick progress updates every 100 episodes
            elif episode % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode:5d} | "
                      f"Reward: {episode_rewards[-1]:8.1f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Time: {elapsed_time:.0f}s")
        
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        print(f"📊 Final metrics:")
        print(f"   Final safety score: {safety_scores[-1]:.1f}%")
        print(f"   Final exploration rate: {self.epsilon:.3f}")
        print(f"   Average Q-value: {np.mean(self.Q_table[self.Q_table != 0]):.1f}")
        print(f"   Non-zero Q-values: {np.sum(self.Q_table != 0):,} / {self.Q_table.size:,}")
        
        # Save training results
        results = {
            'episode_rewards': episode_rewards,
            'safety_scores': safety_scores,
            'exploration_rates': exploration_rates,
            'q_value_evolution': q_value_evolution,
            'training_time': training_time,
            'final_safety_score': safety_scores[-1],
            'hyperparameters': {
                'alpha': self.ALPHA,
                'gamma': self.GAMMA,
                'epsilon_start': 0.3,
                'epsilon_final': self.epsilon,
                'episodes': episodes,
                'episode_samples': episode_samples
            }
        }
        
        return results
    
    def quick_safety_validation(self, episode=None, detailed_logging=False):
        """Quick safety validation using critical scenarios from feature analysis"""
        # Critical test scenarios based on our feature analysis
        test_scenarios = [
            # [c_rate, power, temp, soc, voltage, is_anomaly, expected_safe_actions, description]
            [5.0, 4.0, 4.0, -1.0, 0.0, True, ['pause'], "EXTREME: All critical features at dangerous levels"],
            [-1.0, 1.0, 3.5, -1.0, 0.0, False, ['pause', 'discharge'], "HIGH TEMP: Temperature >99th percentile (78% anomaly rate)"],
            [1.0, 4.0, 1.0, -1.0, 0.0, False, ['pause', 'discharge'], "HIGH POWER: Power >99th percentile (79.5% anomaly rate)"],
            [1.0, 1.0, 1.0, -4.0, -3.0, False, ['slow_charge'], "CRITICAL LOW: SoC <1st percentile (99.8% anomaly rate)"],
            [1.0, 1.0, 1.0, 1.0, 1.0, False, ['discharge', 'maintain'], "HIGH SoC: Battery overcharged"],
            [0.0, 0.0, 0.0, -1.0, 0.0, False, ['slow_charge', 'fast_charge', 'maintain'], "NORMAL: Safe operating conditions"],
            [4.5, 1.0, 1.0, -1.0, 0.0, False, ['pause', 'slow_charge'], "EXTREME C-RATE: >99th percentile (64% anomaly rate)"],
            [1.0, 1.0, 1.0, -1.0, -4.0, False, ['slow_charge', 'fast_charge'], "LOW VOLTAGE: <1st percentile (66.3% anomaly rate)"],
        ]
        
        safe_decisions = 0
        total_scenarios = len(test_scenarios)
        detailed_results = []
        
        if detailed_logging and episode is not None:
            print(f"\n🔍 Detailed Safety Validation - Episode {episode}")
            print("-" * 100)
            print(f"{'Scenario':<35} {'Action':<12} {'Q-Values':<40} {'Safe?':<6} {'Expected'}")
            print("-" * 100)
        
        for i, (c_rate, power, temp, soc, voltage, is_anomaly, safe_actions, description) in enumerate(test_scenarios):
            state = self.discretize_enhanced_state(c_rate, power, temp, soc, voltage, is_anomaly)
            
            # Get all Q-values for this state
            q_values = self.Q_table[state]
            action_idx = np.argmax(q_values)
            chosen_action = self.ACTIONS[action_idx]
            
            is_safe = chosen_action in safe_actions
            if is_safe:
                safe_decisions += 1
            
            if detailed_logging and episode is not None:
                # Format Q-values for display
                q_str = " ".join([f"{self.ACTIONS[j][:4]}:{q_values[j]:6.1f}" for j in range(len(self.ACTIONS))])
                safe_str = "✅" if is_safe else "❌"
                expected_str = "/".join(safe_actions)
                
                print(f"{description[:34]:<35} {chosen_action:<12} {q_str:<40} {safe_str:<6} {expected_str}")
            
            detailed_results.append({
                'scenario': description,
                'state_values': [c_rate, power, temp, soc, voltage, is_anomaly],
                'q_values': q_values.tolist(),
                'chosen_action': chosen_action,
                'expected_actions': safe_actions,
                'is_safe': is_safe
            })
        
        if detailed_logging and episode is not None:
            print("-" * 100)
            print(f"Safety Score: {safe_decisions}/{total_scenarios} = {(safe_decisions/total_scenarios)*100:.1f}%")
            print()
        
        safety_score = (safe_decisions / total_scenarios) * 100
        
        if detailed_logging:
            return safety_score, detailed_results
        else:
            return safety_score
    
    def analyze_q_table_evolution(self, episode):
        """Analyze Q-table evolution and learning progress"""
        non_zero_q = self.Q_table[self.Q_table != 0]
        
        if len(non_zero_q) > 0:
            q_stats = {
                'mean': np.mean(non_zero_q),
                'std': np.std(non_zero_q),
                'min': np.min(non_zero_q),
                'max': np.max(non_zero_q),
                'count': len(non_zero_q),
                'coverage': len(non_zero_q) / self.Q_table.size * 100
            }
            
            print(f"📊 Q-Table Stats (Episode {episode}):")
            print(f"   Non-zero Q-values: {q_stats['count']:,} ({q_stats['coverage']:.1f}% coverage)")
            print(f"   Q-value range: {q_stats['min']:.1f} to {q_stats['max']:.1f}")
            print(f"   Mean Q-value: {q_stats['mean']:.1f} ± {q_stats['std']:.1f}")
            
            return q_stats
        else:
            return None
    
    def log_action_distribution(self, episode, recent_actions):
        """Log distribution of actions taken"""
        if len(recent_actions) > 0:
            action_counts = {action: recent_actions.count(i) for i, action in enumerate(self.ACTIONS)}
            total_actions = len(recent_actions)
            
            print(f"🎯 Action Distribution (Last 1000 actions):")
            for action, count in action_counts.items():
                percentage = (count / total_actions) * 100
                print(f"   {action:12}: {count:4d} ({percentage:5.1f}%)")
        
        return action_counts if len(recent_actions) > 0 else {}
    
    def save_agent(self, agent_name: str, results: dict):
        """Save trained RL agent"""
        # Save Q-table
        q_table_path = self.models_dir / f"{agent_name}_q_table.pkl"
        with open(q_table_path, 'wb') as f:
            pickle.dump(self.Q_table, f)
        
        # Save training results
        results_path = self.models_dir / f"{agent_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save agent configuration
        config = {
            'agent_name': agent_name,
            'state_space_size': self.state_space_size,
            'actions': self.ACTIONS,
            'hyperparameters': results['hyperparameters'],
            'bins': {
                'c_rate_bins': self.C_RATE_BINS.tolist(),
                'power_bins': self.POWER_BINS.tolist(),
                'temp_bins': self.TEMP_BINS.tolist(),
                'soc_bins': self.SOC_BINS.tolist(),
                'voltage_bins': self.VOLTAGE_BINS.tolist()
            },
            'critical_thresholds': self.critical_thresholds
        }
        
        config_path = self.models_dir / f"{agent_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Saved robust RL agent:")
        print(f"   Q-table: {q_table_path}")
        print(f"   Results: {results_path}")
        print(f"   Config: {config_path}")

def main():
    """Main training function"""
    print("🚀 Starting Robust RL Agent Training")
    print("=" * 70)
    
    trainer = RobustRLTrainer()
    
    try:
        # Load training data
        training_data = trainer.load_training_data()
        
        # Train robust agent
        results = trainer.train_robust_agent(
            training_data, 
            episodes=15000,  # More episodes for better learning
            episode_samples=1000  # More samples per episode
        )
        
        # Save the trained agent
        trainer.save_agent("rl_robust_enhanced", results)
        
        print(f"\n🎉 Robust RL Agent Training Complete!")
        print("=" * 70)
        print(f"🏆 Final Performance:")
        print(f"   Safety Score: {results['final_safety_score']:.1f}%")
        print(f"   Training Time: {results['training_time']:.1f} seconds")
        print(f"   Episodes: {results['hyperparameters']['episodes']:,}")
        print(f"   Total Interactions: {results['hyperparameters']['episodes'] * results['hyperparameters']['episode_samples']:,}")
        
        print(f"\n✅ Ready for integration with dashboard!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
