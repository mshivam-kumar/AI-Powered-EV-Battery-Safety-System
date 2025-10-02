#!/usr/bin/env python3
"""
RL Agent Training - Separate Script
Train Q-Learning agent for battery management decisions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLTrainer:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # RL Configurations to test (FIXED - Safety-First Training)
        self.rl_configs = {
            'rl_safety_focused': {
                'name': 'Safety-Focused RL',
                'alpha': 0.2,  # Higher learning rate for faster convergence
                'gamma': 0.95,  # Good future planning
                'epsilon': 0.3,  # Higher exploration to find safe actions
                'epsilon_decay': 0.99,
                'episodes': 1500,  # Moderate training time
                'episode_samples': 200
            },
            'rl_conservative': {
                'name': 'Conservative RL',
                'alpha': 0.15,
                'gamma': 0.98,  # Very future-focused
                'epsilon': 0.2,
                'epsilon_decay': 0.995,
                'episodes': 2000,
                'episode_samples': 150
            }
        }
        
        # State and Action spaces (consistent across all configs)
        self.SOC_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        self.TEMP_BINS = [0.0, 20.0, 30.0, 40.0, 50.0, 100.0]
        self.AMBIENT_BINS = [0.0, 20.0, 30.0, 40.0, 60.0, 100.0]
        self.VOLTAGE_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
        self.ACTIONS = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
        
    def load_data(self, data_dir: str = "data"):
        """Load training data for RL"""
        data_path = Path(data_dir)
        
        # Load features
        features_file = data_path / "processed" / "features" / "extracted_features.parquet"
        features_df = pd.read_parquet(features_file)
        
        # Load labels
        labels_file = data_path / "processed" / "labels" / "consensus_labels.npy"
        labels = np.load(labels_file)
        
        # Extract features
        feature_columns = [col for col in features_df.columns 
                          if col not in ['battery_id', 'battery_type', 'time']]
        X = features_df[feature_columns].values
        
        # Use only training portion for RL (same split as other models)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, 
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        print(f"📊 RL Training Data:")
        print(f"   • Training samples: {len(X_train):,}")
        print(f"   • Features: {X_train.shape[1]}")
        print(f"   • Anomaly rate: {np.mean(y_train == 1):.1%}")
        
        return X_train, y_train
    
    def discretize_state(self, soc, temp, ambient, voltage):
        """Discretize continuous state to discrete bins"""
        n_soc_bins = len(self.SOC_BINS) - 1
        n_temp_bins = len(self.TEMP_BINS) - 1
        n_ambient_bins = len(self.AMBIENT_BINS) - 1
        n_voltage_bins = len(self.VOLTAGE_BINS) - 1
        
        soc_bin = min(max(int(soc * n_soc_bins), 0), n_soc_bins - 1)
        temp_bin = min(max(int(temp * n_temp_bins), 0), n_temp_bins - 1)
        ambient_bin = min(max(int(ambient * n_ambient_bins), 0), n_ambient_bins - 1)
        voltage_bin = min(max(int(voltage * n_voltage_bins), 0), n_voltage_bins - 1)
        
        return soc_bin, temp_bin, ambient_bin, voltage_bin
    
    def compute_reward(self, soc, temp, ambient, voltage, is_anomaly, action_idx):
        """CORRECTED: Safety-First Reward Function with Proper Thresholds"""
        action = self.ACTIONS[action_idx]
        
        # CORRECTED THRESHOLDS based on standardized data:
        # Temperature: 95th percentile = 2.167, 90th = 1.429
        # SoC: mostly negative values, 10th percentile = 0.039 (high SoC)
        # High temp = > 1.5, Very high temp = > 2.0
        # Low SoC = < -5.0, Very low SoC = < -10.0
        # High SoC = > 0.0 (since most values are negative)
        
        # START WITH MASSIVE NEGATIVE PENALTY FOR UNSAFE ACTIONS
        reward = -1000.0  # Start with huge penalty, earn way back up
        
        # ABSOLUTE SAFETY VIOLATIONS (GAME OVER)
        if temp > 2.0 and action == 'fast_charge':
            return -10000.0  # INSTANT MASSIVE PENALTY - NEVER DO THIS!
        
        if soc > 0.0 and action in ['fast_charge', 'slow_charge']:
            return -5000.0  # NEVER charge when battery is full!
        
        if is_anomaly and action in ['fast_charge', 'slow_charge']:
            return -8000.0  # NEVER charge during anomalies!
        
        # HIGH TEMPERATURE SAFETY (Still very dangerous)
        if temp > 1.5:  # High temperature (90th+ percentile)
            if action == 'pause':
                reward = 500.0  # HUGE reward for pausing when hot
            elif action == 'discharge':
                reward = 300.0  # Good to discharge when hot
            elif action == 'maintain':
                reward = 100.0  # Okay to maintain
            elif action == 'slow_charge':
                reward = -2000.0  # Still very bad to charge when hot
            return reward
        
        # ANOMALY SAFETY
        if is_anomaly:
            if action == 'pause':
                reward = 400.0  # HUGE reward for pausing during anomaly
            elif action == 'discharge':
                reward = 200.0  # Good to discharge
            elif action == 'maintain':
                reward = 50.0  # Okay to maintain
            return reward
        
        # CRITICAL SoC MANAGEMENT
        if soc < -10.0:  # CRITICALLY low - must charge safely
            if action == 'slow_charge' and temp < 1.0:
                reward = 300.0  # Safe emergency charging
            elif action == 'fast_charge' and temp < 0.0:
                reward = 200.0  # Fast charge only if very cool
            elif action == 'pause':
                reward = -3000.0  # DON'T pause when critically low!
            elif action in ['discharge', 'maintain']:
                reward = -1000.0  # Don't discharge when already low
            return reward
        
        if soc > 0.0:  # CRITICALLY high - must not charge
            if action == 'discharge':
                reward = 400.0  # EXCELLENT to discharge when full
            elif action == 'maintain':
                reward = 200.0  # Good to maintain
            elif action == 'pause':
                reward = 100.0  # Okay to pause
            # Charging actions already handled above with massive penalties
            return reward
        
        # NORMAL OPERATION (-10.0 <= soc <= 0.0, temp <= 1.5, no anomaly)
        reward = 0  # Reset to neutral for normal conditions
        
        # Temperature-based safety (using corrected thresholds)
        if temp <= 0.0:  # Very cool - safest conditions
            if action == 'fast_charge' and soc < -2.0:
                reward += 100.0  # Safe fast charging
            elif action == 'slow_charge':
                reward += 80.0  # Always safe
            elif action == 'maintain' and soc > -5.0:
                reward += 60.0  # Good maintenance
        elif temp <= 1.0:  # Normal temperature - moderately safe
            if action == 'slow_charge':
                reward += 50.0  # Prefer slow charging
            elif action == 'fast_charge' and soc < -5.0:
                reward += 20.0  # Fast charge only when needed
            elif action == 'maintain' and soc > -5.0:
                reward += 40.0  # Good maintenance
        else:  # temp > 1.0 but <= 1.5 - getting warm, be careful
            if action == 'slow_charge' and soc < -5.0:
                reward += 20.0  # Careful slow charging
            elif action == 'pause':
                reward += 30.0  # Good to pause when warming
            elif action == 'maintain':
                reward += 25.0  # Safe maintenance
            elif action == 'fast_charge':
                reward -= 100.0  # Discourage fast charging when warm
        
        # SoC optimization (corrected range)
        if -8.0 <= soc <= -2.0:  # Optimal range
            reward += 20.0  # Bonus for being in good range
        
        return reward
    
    def quick_safety_check(self, Q_table, episode):
        """Quick safety validation during training"""
        # Critical test scenarios for safety validation (CORRECTED THRESHOLDS)
        test_scenarios = [
            # [soc, temp, ambient, voltage, expected_safe_actions]
            [-5.0, 2.0, 1.0, 0.0, ['pause']],  # Low SoC + High Temp = PAUSE
            [0.01, 2.0, 1.0, 1.0, ['pause', 'discharge']],  # High SoC + High Temp = PAUSE/DISCHARGE
            [-15.0, 0.5, 0.0, 0.0, ['slow_charge']],  # Critical Low SoC = SLOW CHARGE
            [0.01, 1.8, 1.0, 1.0, ['pause', 'discharge']],  # Critical High SoC + Temp = PAUSE/DISCHARGE
        ]
        
        safe_decisions = 0
        total_scenarios = len(test_scenarios)
        
        for soc, temp, ambient, voltage, safe_actions in test_scenarios:
            state = self.discretize_state(soc, temp, ambient, voltage)
            action_idx = np.argmax(Q_table[state])
            chosen_action = self.ACTIONS[action_idx]
            
            if chosen_action in safe_actions:
                safe_decisions += 1
        
        return safe_decisions / total_scenarios
    
    def train_single_rl(self, config_name: str, config: dict, X_train, y_train):
        """Train a single RL configuration"""
        print(f"\n🤖 Training {config['name']}...")
        print(f"   Learning Rate: {config['alpha']}")
        print(f"   Discount Factor: {config['gamma']}")
        print(f"   Episodes: {config['episodes']}")
        print(f"   Exploration Rate: {config['epsilon']} (decay: {config['epsilon_decay']})")
        
        # Initialize Q-table
        n_soc_bins = len(self.SOC_BINS) - 1
        n_temp_bins = len(self.TEMP_BINS) - 1
        n_ambient_bins = len(self.AMBIENT_BINS) - 1
        n_voltage_bins = len(self.VOLTAGE_BINS) - 1
        n_actions = len(self.ACTIONS)
        
        Q_table = np.zeros((n_soc_bins, n_temp_bins, n_ambient_bins, n_voltage_bins, n_actions))
        
        print(f"   Q-table shape: {Q_table.shape}")
        print(f"   Total states: {np.prod(Q_table.shape[:-1]):,}")
        
        # Training parameters
        alpha = config['alpha']
        gamma = config['gamma']
        epsilon = config['epsilon']
        epsilon_decay = config['epsilon_decay']
        n_episodes = config['episodes']
        episode_samples = config['episode_samples']
        
        # Training loop
        episode_rewards = []
        exploration_rates = []
        
        print(f"   🔄 Training for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            episode_reward = 0
            
            # Sample random subset for this episode
            n_samples = min(episode_samples, len(X_train))
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            
            for i in indices:
                sample = X_train[i]
                is_anomaly = y_train[i] == 1  # 1 = anomaly, 0 = normal
                
                # Extract and normalize state variables
                soc = max(0, min(1, (sample[3] + 3) / 6))        # SoC
                temp = max(0, min(1, (sample[2] + 3) / 6))       # Temperature
                ambient = max(0, min(1, (sample[4] + 3) / 6))    # Ambient temp
                voltage = max(0, min(1, (sample[0] + 3) / 6))    # Voltage
                
                # Discretize state
                state = self.discretize_state(soc, temp, ambient, voltage)
                
                # Choose action (epsilon-greedy)
                if np.random.random() < epsilon:
                    action = np.random.choice(n_actions)
                else:
                    action = np.argmax(Q_table[state])
                
                # Compute reward
                reward = self.compute_reward(soc, temp, ambient, voltage, is_anomaly, action)
                episode_reward += reward
                
                # Update Q-table (Q-learning)
                old_value = Q_table[state][action]
                max_future_q = np.max(Q_table[state])
                new_value = old_value + alpha * (reward + gamma * max_future_q - old_value)
                Q_table[state][action] = new_value
            
            episode_rewards.append(episode_reward)
            exploration_rates.append(epsilon)
            
            # Decay exploration
            epsilon = max(0.01, epsilon * epsilon_decay)
            
            # Progress logging with safety validation
            if episode % (n_episodes // 10) == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                print(f"      Episode {episode:4d}: Avg Reward = {avg_reward:6.2f}, Epsilon = {epsilon:.3f}")
                
                # SAFETY CHECK: Test agent behavior in critical scenarios
                if episode > 0 and episode % (n_episodes // 5) == 0:  # Every 20% of training
                    safety_score = self.quick_safety_check(Q_table, episode)
                    print(f"         🛡️  Safety Score: {safety_score:.1%} (Episode {episode})")
        
        # Final statistics
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        total_states_visited = np.sum(Q_table != 0)
        state_coverage = total_states_visited / np.prod(Q_table.shape)
        
        results = {
            'config_name': config_name,
            'name': config['name'],
            'alpha': config['alpha'],
            'gamma': config['gamma'],
            'initial_epsilon': config['epsilon'],
            'epsilon_decay': config['epsilon_decay'],
            'final_epsilon': epsilon,
            'n_episodes': config['episodes'],
            'episode_samples': config['episode_samples'],
            'final_avg_reward': float(final_avg_reward),
            'total_reward': float(np.sum(episode_rewards)),
            'states_explored': int(total_states_visited),
            'state_coverage': float(state_coverage),
            'q_table_shape': Q_table.shape,
            'convergence_episode': len(episode_rewards)
        }
        
        # FINAL SAFETY VALIDATION
        final_safety_score = self.quick_safety_check(Q_table, n_episodes)
        
        print(f"   ✅ Results:")
        print(f"      • Final Avg Reward: {final_avg_reward:.3f}")
        print(f"      • Final Safety Score: {final_safety_score:.1%} {'✅ SAFE' if final_safety_score >= 0.75 else '⚠️ UNSAFE'}")
        print(f"      • States Explored: {total_states_visited:,} / {np.prod(Q_table.shape):,} ({state_coverage:.1%})")
        print(f"      • Final Epsilon: {epsilon:.3f}")
        
        # Save Q-table
        q_table_file = self.models_dir / f"{config_name}_q_table.pkl"
        with open(q_table_file, 'wb') as f:
            pickle.dump(Q_table, f)
        
        # Save training history
        history = {
            'episode_rewards': episode_rewards,
            'exploration_rates': exploration_rates,
            'config': config
        }
        history_file = self.models_dir / f"{config_name}_history.pkl"
        with open(history_file, 'wb') as f:
            pickle.dump(history, f)
        
        print(f"      • Saved Q-table: {q_table_file}")
        print(f"      • Saved history: {history_file}")
        
        return Q_table, results, episode_rewards
    
    def plot_rl_comparison(self, all_results, all_rewards):
        """Create comparison plots for all RL agents"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        names = [r['name'] for r in all_results]
        final_rewards = [r['final_avg_reward'] for r in all_results]
        state_coverage = [r['state_coverage'] for r in all_results]
        episodes = [r['n_episodes'] for r in all_results]
        
        # Final Average Rewards
        ax1.bar(names, final_rewards, color='skyblue', alpha=0.7)
        ax1.set_title('Final Average Reward')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # State Coverage
        ax2.bar(names, [c * 100 for c in state_coverage], color='lightgreen', alpha=0.7)
        ax2.set_title('State Space Coverage')
        ax2.set_ylabel('Coverage (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Training Episodes
        ax3.bar(names, episodes, color='salmon', alpha=0.7)
        ax3.set_title('Training Episodes')
        ax3.set_ylabel('Episodes')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Reward Evolution
        for i, (name, rewards) in enumerate(zip(names, all_rewards)):
            # Smooth rewards for better visualization
            window = max(1, len(rewards) // 50)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax4.plot(smoothed, label=name, alpha=0.8)
        
        ax4.set_title('Reward Evolution During Training')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.models_dir / "rl_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 RL comparison plot saved: {plot_file}")
    
    def run_rl_training(self):
        """Train all RL configurations and compare"""
        print("🤖 RL Agent Training")
        print("=" * 50)
        
        # Load data
        X_train, y_train = self.load_data()
        
        # Train all configurations
        trained_agents = {}
        all_results = []
        all_rewards = []
        
        for config_name, config in self.rl_configs.items():
            q_table, results, rewards = self.train_single_rl(
                config_name, config, X_train, y_train
            )
            trained_agents[config_name] = q_table
            all_results.append(results)
            all_rewards.append(rewards)
        
        # Find best agent
        best_agent = max(all_results, key=lambda x: x['final_avg_reward'])
        
        print(f"\n🏆 Best RL Agent: {best_agent['name']}")
        print(f"   • Final Avg Reward: {best_agent['final_avg_reward']:.3f}")
        print(f"   • State Coverage: {best_agent['state_coverage']:.1%}")
        print(f"   • Learning Rate: {best_agent['alpha']}")
        print(f"   • Episodes: {best_agent['n_episodes']}")
        
        # Save results
        results_file = self.models_dir / "rl_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': all_results,
                'best_agent': best_agent,
                'state_action_info': {
                    'soc_bins': self.SOC_BINS,
                    'temp_bins': self.TEMP_BINS,
                    'ambient_bins': self.AMBIENT_BINS,
                    'voltage_bins': self.VOLTAGE_BINS,
                    'actions': self.ACTIONS
                },
                'summary': {
                    'total_agents': len(all_results),
                    'best_config': best_agent['config_name'],
                    'best_reward': best_agent['final_avg_reward'],
                    'best_coverage': best_agent['state_coverage']
                }
            }, f, indent=2)
        
        # Create comparison plot
        self.plot_rl_comparison(all_results, all_rewards)
        
        # Summary table
        print(f"\n📊 RL Agents Summary:")
        print("-" * 90)
        print(f"{'Agent':<15} {'Reward':<10} {'Coverage':<10} {'Episodes':<10} {'Alpha':<8} {'Gamma':<8}")
        print("-" * 90)
        for r in all_results:
            print(f"{r['config_name']:<15} {r['final_avg_reward']:<10.3f} {r['state_coverage']:<10.1%} {r['n_episodes']:<10} {r['alpha']:<8.2f} {r['gamma']:<8.2f}")
        
        print(f"\n💾 Results saved:")
        print(f"   • Q-tables: {self.models_dir}/*_q_table.pkl")
        print(f"   • Results: {results_file}")
        print(f"   • Plot: {self.models_dir}/rl_comparison.png")
        
        return trained_agents, all_results, best_agent

def main():
    """Main function"""
    trainer = RLTrainer()
    agents, results, best = trainer.run_rl_training()
    
    print(f"\n✅ RL training completed!")
    print(f"🏆 Best agent: {best['name']} (Reward: {best['final_avg_reward']:.3f})")

if __name__ == "__main__":
    main()

