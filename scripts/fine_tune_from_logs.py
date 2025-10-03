#!/usr/bin/env python3
"""
Fine-tune RL Agent using untrained states from logs
Uses the rl_untrained_states.json file to improve coverage
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime

class LogBasedFineTuner:
    def __init__(self, agent_path: str = 'models/rl_robust_enhanced_v2_q_table.pkl'):
        """Initialize fine-tuner with existing agent"""
        self.agent_path = agent_path
        
        # Actions (initialize first)
        self.actions = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
        
        # RL parameters for fine-tuning
        self.alpha = 0.05  # Lower learning rate for fine-tuning
        self.gamma = 0.9
        self.epsilon = 0.3  # Lower exploration for fine-tuning
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        
        # Load agent
        self.load_agent()
        
        print(f"✅ Fine-tuner initialized")
        print(f"📊 Agent: {agent_path}")
        print(f"📊 Q-table shape: {self.q_table.shape}")
        print(f"📊 Actions: {len(self.actions)}")
    
    def load_agent(self):
        """Load existing RL agent"""
        try:
            if self.agent_path.endswith('.json'):
                # Load JSON format
                with open(self.agent_path, 'r') as f:
                    agent_data = json.load(f)
                self.q_table = np.array(agent_data['q_table'])
                self.actions = agent_data.get('actions', self.actions)
                self.state_dims = agent_data.get('state_dims', [5, 5, 5, 4])
            else:
                # Load pickle format
                import pickle
                with open(self.agent_path, 'rb') as f:
                    self.q_table = pickle.load(f)
                # Infer state dimensions from Q-table shape
                self.state_dims = list(self.q_table.shape[:-1])  # All dimensions except last (actions)
            
            print(f"✅ Loaded agent from {self.agent_path}")
            print(f"📊 Q-table shape: {self.q_table.shape}")
            print(f"📊 State dimensions: {self.state_dims}")
            
        except Exception as e:
            print(f"❌ Failed to load agent: {e}")
            raise
    
    def load_untrained_states(self, log_path: str = 'rl_untrained_states.json'):
        """Load untrained states from logs"""
        try:
            with open(log_path, 'r') as f:
                untrained_data = json.load(f)
            
            print(f"📊 Loaded {len(untrained_data)} untrained states from {log_path}")
            
            # Extract state information
            states = []
            for entry in untrained_data:
                state_info = {
                    'state_bins': entry['state_bins'],
                    'state_index': entry['state_index'],
                    'standardized_telemetry': entry['standardized_telemetry'],
                    'real_world_context': entry['real_world_context'],
                    'safety_priority': entry.get('safety_priority', 'normal'),
                    'q_values_zero': entry.get('q_values_zero', [0.0] * len(self.actions))
                }
                states.append(state_info)
            
            return states
            
        except Exception as e:
            print(f"❌ Failed to load untrained states: {e}")
            return []
    
    def generate_scenarios_from_logs(self, untrained_states: List[Dict], num_scenarios: int = 1000):
        """Generate training scenarios from untrained states"""
        scenarios = []
        
        for _ in range(num_scenarios):
            # Randomly select an untrained state
            base_state = np.random.choice(untrained_states)
            
            # Extract base values
            std_telemetry = base_state['standardized_telemetry']
            real_context = base_state['real_world_context']
            
            # Add some variation to create diverse scenarios
            variation_factor = np.random.uniform(0.8, 1.2)
            
            scenario = {
                'soc': max(0.0, min(1.0, std_telemetry['soc'] * variation_factor)),
                'temperature': max(0.0, min(1.0, std_telemetry['temperature'] * variation_factor)),
                'ambient_temp': max(0.0, min(1.0, std_telemetry['temperature'] * variation_factor * 0.9)),
                'voltage': max(0.0, min(1.0, std_telemetry['voltage'] * variation_factor)),
                'c_rate': std_telemetry.get('c_rate', 0.0) * variation_factor,
                'power': std_telemetry.get('power', 0.0) * variation_factor,
                'is_anomaly': std_telemetry['is_anomaly'],
                'safety_priority': base_state['safety_priority'],
                'scenario_type': 'log_based_fine_tuning'
            }
            
            scenarios.append(scenario)
        
        print(f"✅ Generated {len(scenarios)} scenarios from untrained states")
        return scenarios
    
    def compute_fine_tuning_reward(self, soc, temp, ambient, voltage, is_anomaly, action_idx, safety_priority):
        """Compute reward for fine-tuning with safety priority"""
        action = self.actions[action_idx]
        reward = 0.0
        
        # Base safety rewards
        if is_anomaly:
            if action == 'pause':
                reward += 100.0
            elif action == 'slow_charge':
                reward += 50.0
            elif action == 'fast_charge':
                reward -= 200.0
            elif action == 'maintain':
                reward -= 100.0
        else:
            if action == 'maintain':
                reward += 50.0
            elif action == 'slow_charge':
                reward += 30.0
            elif action == 'fast_charge':
                reward += 20.0
            elif action == 'pause':
                reward -= 20.0
        
        # Safety priority adjustments
        if safety_priority == 'high':
            # High priority states get extra rewards for safe actions
            if action == 'pause' and (temp > 0.8 or soc < 0.2):
                reward += 50.0  # Extra reward for safe action in high priority state
            elif action == 'slow_charge' and soc < 0.3:
                reward += 30.0  # Extra reward for charging in low SoC high priority state
        
        # Temperature-based rewards
        if temp > 0.8:
            if action == 'pause':
                reward += 50.0
            elif action == 'fast_charge':
                reward -= 100.0
        elif temp < 0.2:
            if action == 'slow_charge':
                reward += 40.0
            elif action == 'fast_charge':
                reward -= 60.0
        
        # SoC-based rewards
        if soc < 0.2:
            if action in ['slow_charge', 'fast_charge']:
                reward += 60.0
            elif action == 'pause':
                reward -= 40.0
        elif soc > 0.8:
            if action in ['pause', 'maintain']:
                reward += 50.0
            elif action == 'fast_charge':
                reward -= 80.0
        
        return reward
    
    def fine_tune_agent(self, untrained_states: List[Dict], episodes: int = 1000):
        """Fine-tune agent using untrained states"""
        print(f"🎯 Starting fine-tuning with {len(untrained_states)} untrained states")
        print(f"📊 Episodes: {episodes}")
        print("=" * 60)
        
        # Generate scenarios from untrained states
        scenarios = self.generate_scenarios_from_logs(untrained_states, len(untrained_states) * 2)
        
        episode_rewards = []
        state_coverage = set()
        
        start_time = time.time()
        
        for episode in range(episodes):
            episode_reward = 0
            
            # Sample scenarios for this episode
            episode_scenarios = np.random.choice(scenarios, min(100, len(scenarios)), replace=False)
            
            for scenario in episode_scenarios:
                # Extract state variables
                soc = scenario['soc']
                temp = scenario['temperature']
                ambient = scenario['ambient_temp']
                voltage = scenario['voltage']
                is_anomaly = scenario['is_anomaly']
                safety_priority = scenario['safety_priority']
                
                # Discretize state (6D)
                # Extract additional features from scenario
                c_rate = scenario.get('c_rate', 0.0)
                power = scenario.get('power', 0.0)
                
                state = self.discretize_state(soc, temp, ambient, voltage, c_rate, power, is_anomaly)
                state_coverage.add(state)
                
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action_idx = np.random.randint(len(self.actions))
                else:
                    action_idx = np.argmax(self.q_table[state])
                
                # Compute fine-tuning reward
                reward = self.compute_fine_tuning_reward(soc, temp, ambient, voltage, is_anomaly, action_idx, safety_priority)
                episode_reward += reward
                
                # Q-learning update
                old_value = self.q_table[state][action_idx]
                max_future_q = np.max(self.q_table[state])
                new_value = old_value + self.alpha * (reward + self.gamma * max_future_q - old_value)
                self.q_table[state][action_idx] = new_value
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Progress reporting
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                coverage = len(state_coverage) / np.prod(self.state_dims) * 100
                print(f"📊 Episode {episode:4d} | Coverage: {coverage:5.1f}% | Epsilon: {self.epsilon:.3f} | Avg Reward: {avg_reward:8.1f} | States: {len(state_coverage)}")
        
        training_time = time.time() - start_time
        
        print(f"\n🎯 FINE-TUNING COMPLETED!")
        print(f"📊 Final Coverage: {len(state_coverage)/np.prod(self.state_dims)*100:.1f}%")
        print(f"📊 States Visited: {len(state_coverage)}")
        print(f"📊 Final Avg Reward: {np.mean(episode_rewards[-100:]):.1f}")
        print(f"📊 Training Time: {training_time:.1f} seconds")
        
        return episode_rewards, state_coverage
    
    def discretize_state(self, soc, temp, ambient, voltage, c_rate=0.0, power=0.0, is_anomaly=False):
        """Discretize state for 6D state space (matching dashboard)"""
        # Calculate derived features (matching dashboard logic)
        c_rate = abs(c_rate)
        power = abs(power)
        
        # Discretize to bins (matching dashboard: 5x5x5x5x5x2)
        c_rate_bin = min(max(int(c_rate * 2.0), 0), 4)  # 0-4 bins
        power_bin = min(max(int(power / 2.0), 0), 4)     # 0-4 bins  
        temp_bin = min(max(int((temp + 2.0) / 1.0), 0), 4)  # 0-4 bins
        soc_bin = min(max(int((soc + 10.0) / 4.0), 0), 4)  # 0-4 bins
        voltage_bin = min(max(int((voltage + 2.0) / 1.0), 0), 4)  # 0-4 bins
        
        # Anomaly flag (0 or 1)
        anomaly_bin = 1 if is_anomaly else 0
        
        return (c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin)
    
    def save_fine_tuned_agent(self, output_path: str = 'models/fine_tuned_from_logs_rl_agent.json'):
        """Save the fine-tuned agent and clean untrained states log"""
        agent_data = {
            'q_table': self.q_table.tolist(),
            'actions': self.actions,
            'state_dims': self.state_dims,
            'agent_type': 'fine_tuned_from_logs_rl_agent',
            'training_time': datetime.now().isoformat(),
            'fine_tuning_params': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(agent_data, f, indent=2)
        
        # Clean untrained states log after successful fine-tuning
        # This is the ONLY place where the counter gets reset
        untrained_log_path = 'rl_untrained_states.json'
        if Path(untrained_log_path).exists():
            # Backup the used states before cleaning
            backup_path = f'rl_untrained_states_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(untrained_log_path, 'r') as f:
                used_states = json.load(f)
            with open(backup_path, 'w') as f:
                json.dump(used_states, f, indent=2)
            
            # Clear the log for fresh untrained states
            with open(untrained_log_path, 'w') as f:
                json.dump([], f)
                f.flush()  # Ensure data is written
            
            # Verify the file was written correctly
            with open(untrained_log_path, 'r') as f:
                content = f.read().strip()
                if content == "[]":
                    print(f"✅ Verified: {untrained_log_path} contains proper empty array")
                else:
                    print(f"⚠️ Warning: {untrained_log_path} content: {content}")
            
            print(f"🧹 Cleaned {untrained_log_path} after fine-tuning")
            print(f"📁 Backup saved to {backup_path}")
            print(f"📊 Reset untrained states counter to 0 (automatic after fine-tuning)")
            print(f"🔄 Dashboard will now show 0 unique states")
        
        print(f"💾 Fine-tuned agent saved to {output_path}")

def main():
    """Main fine-tuning function"""
    print("🎯 RL Agent Fine-tuning from Logs")
    print("=" * 50)
    
    # Initialize fine-tuner
    fine_tuner = LogBasedFineTuner()
    
    # Load untrained states
    untrained_states = fine_tuner.load_untrained_states()
    
    if not untrained_states:
        print("❌ No untrained states found!")
        return
    
    # Fine-tune agent
    episode_rewards, state_coverage = fine_tuner.fine_tune_agent(untrained_states, episodes=1000)
    
    # Save fine-tuned agent
    fine_tuner.save_fine_tuned_agent()
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"📈 Final coverage: {len(state_coverage)/np.prod(fine_tuner.state_dims)*100:.1f}%")
    print(f"🎯 Fine-tuned agent ready for deployment")

if __name__ == "__main__":
    main()
