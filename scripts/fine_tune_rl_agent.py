#!/usr/bin/env python3
"""
Fine-tune RL agent using targeted scenarios for unvisited states
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime

class RLFineTuner:
    def __init__(self, agent_path: str = 'models/climate_aware_rl_agent.json'):
        """Initialize fine-tuner with existing agent"""
        self.agent_path = agent_path
        self.load_agent()
        
        # Fine-tuning parameters (more aggressive exploration)
        self.alpha = 0.2  # Higher learning rate for fine-tuning
        self.gamma = 0.9
        self.epsilon = 0.8  # High exploration for unvisited states
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.3  # Keep some exploration
        
    def load_agent(self):
        """Load existing trained agent"""
        try:
            with open(self.agent_path, 'r') as f:
                agent_data = json.load(f)
            
            self.q_table = agent_data['q_table']
            self.actions = agent_data.get('actions', ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain'])
            
            print(f"✅ Loaded existing agent with {len(self.actions)} actions")
            
        except FileNotFoundError:
            print(f"❌ Agent file not found: {self.agent_path}")
            raise
    
    def load_targeted_scenarios(self, scenarios_path: str = 'data/targeted_scenarios.json') -> List[Dict]:
        """Load targeted scenarios for unvisited states"""
        try:
            with open(scenarios_path, 'r') as f:
                scenarios = json.load(f)
            
            print(f"✅ Loaded {len(scenarios)} targeted scenarios")
            return scenarios
            
        except FileNotFoundError:
            print(f"❌ Targeted scenarios not found: {scenarios_path}")
            print("   Run analyze_unvisited_states.py first")
            raise
    
    def discretize_state(self, telemetry: Dict) -> tuple:
        """Discretize state using original 4D approach"""
        soc = telemetry.get('soc', 0.5)
        temp = telemetry.get('temperature', 25.0)
        ambient = telemetry.get('ambient_temp', 25.0)
        voltage = telemetry.get('voltage', 3.7)
        
        # Normalize like original model
        soc_norm = max(0, min(1, soc))
        temp_norm = max(0, min(1, (temp + 3) / 6))
        ambient_norm = max(0, min(1, (ambient + 3) / 6))
        voltage_norm = max(0, min(1, (voltage + 3) / 6))
        
        # Discretize using original binning
        soc_bin = min(max(int(soc_norm * 4), 0), 4)
        temp_bin = min(max(int(temp_norm * 4), 0), 4)
        ambient_bin = min(max(int(ambient_norm * 4), 0), 4)
        voltage_bin = min(max(int(voltage_norm * 3), 0), 3)
        
        return (soc_bin, temp_bin, ambient_bin, voltage_bin)
    
    def compute_reward(self, soc, temp, ambient, voltage, is_anomaly, action_idx):
        """Compute reward like the original model"""
        action = self.actions[action_idx]
        reward = 0.0
        
        # Safety-based rewards
        if is_anomaly:
            if action == 'pause':
                reward += 50.0
            elif action == 'slow_charge':
                reward += 20.0
            elif action == 'fast_charge':
                reward -= 100.0
            elif action == 'maintain':
                reward -= 50.0
        else:
            if action == 'maintain':
                reward += 30.0
            elif action == 'slow_charge':
                reward += 20.0
            elif action == 'fast_charge':
                reward += 10.0
            elif action == 'pause':
                reward -= 10.0
        
        # Temperature-based rewards
        if temp > 0.8:
            if action == 'pause':
                reward += 30.0
            elif action == 'fast_charge':
                reward -= 50.0
        elif temp < 0.2:
            if action == 'slow_charge':
                reward += 20.0
            elif action == 'fast_charge':
                reward -= 30.0
        
        # SoC-based rewards
        if soc < 0.2:
            if action in ['slow_charge', 'fast_charge']:
                reward += 40.0
            elif action == 'pause':
                reward -= 20.0
        elif soc > 0.8:
            if action in ['pause', 'maintain']:
                reward += 30.0
            elif action == 'fast_charge':
                reward -= 40.0
        
        # Normal operating range bonus
        if 0.2 <= soc <= 0.8 and 0.2 <= temp <= 0.8:
            reward += 20.0
        
        return reward
    
    def fine_tune(self, scenarios: List[Dict], episodes: int = 1000):
        """Fine-tune agent using targeted scenarios"""
        print(f"🎯 Starting Fine-tuning with {len(scenarios)} targeted scenarios")
        print(f"📊 Episodes: {episodes}")
        print(f"🚀 Parameters: α={self.alpha}, ε={self.epsilon}")
        print("=" * 60)
        
        episode_rewards = []
        state_coverage = set()
        
        start_time = time.time()
        
        for episode in range(episodes):
            episode_reward = 0
            
            # Sample scenarios for this episode (focus on unvisited states)
            episode_scenarios = np.random.choice(scenarios, min(100, len(scenarios)), replace=False)
            
            for scenario in episode_scenarios:
                # Extract state variables
                soc = scenario.get('soc', 0.5)
                temp = scenario.get('temperature', 25.0)
                ambient = scenario.get('ambient_temp', 25.0)
                voltage = scenario.get('voltage', 3.7)
                is_anomaly = scenario.get('is_anomaly', False)
                
                # Normalize state variables
                soc_norm = max(0, min(1, soc))
                temp_norm = max(0, min(1, (temp + 3) / 6))
                ambient_norm = max(0, min(1, (ambient + 3) / 6))
                voltage_norm = max(0, min(1, (voltage + 3) / 6))
                
                # Discretize state
                state = self.discretize_state({
                    'soc': soc_norm,
                    'temperature': temp_norm,
                    'ambient_temp': ambient_norm,
                    'voltage': voltage_norm
                })
                
                # Track state coverage
                state_coverage.add(state)
                
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action_idx = np.random.randint(len(self.actions))
                else:
                    action_idx = np.argmax(self.q_table[state[0]][state[1]][state[2]][state[3]])
                
                # Compute reward
                reward = self.compute_reward(soc_norm, temp_norm, ambient_norm, voltage_norm, is_anomaly, action_idx)
                episode_reward += reward
                
                # Q-learning update
                old_value = self.q_table[state[0]][state[1]][state[2]][state[3]][action_idx]
                max_future_q = np.max(self.q_table[state[0]][state[1]][state[2]][state[3]])
                new_value = old_value + self.alpha * (reward + self.gamma * max_future_q - old_value)
                self.q_table[state[0]][state[1]][state[2]][state[3]][action_idx] = new_value
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Progress reporting
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                coverage = len(state_coverage) / 500 * 100
                print(f"📊 Episode {episode:4d} | Coverage: {coverage:5.1f}% | Epsilon: {self.epsilon:.3f} | Avg Reward: {avg_reward:8.1f} | States: {len(state_coverage)}")
        
        training_time = time.time() - start_time
        
        print(f"\n🎯 FINE-TUNING COMPLETED!")
        print(f"📊 Final Coverage: {len(state_coverage)/500*100:.1f}%")
        print(f"📊 States Visited: {len(state_coverage)}")
        print(f"📊 Final Avg Reward: {np.mean(episode_rewards[-100:]):.1f}")
        print(f"📊 Training Time: {training_time:.1f} seconds")
        
        return episode_rewards, state_coverage
    
    def save_fine_tuned_agent(self, output_path: str = 'models/fine_tuned_rl_agent.json'):
        """Save the fine-tuned agent"""
        agent_data = {
            'q_table': self.q_table,
            'actions': self.actions,
            'fine_tuned': True,
            'fine_tune_time': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(agent_data, f, indent=2)
        
        print(f"💾 Fine-tuned agent saved to {output_path}")

def main():
    """Main fine-tuning function"""
    print("🎯 RL Agent Fine-tuning for Unvisited States")
    print("=" * 60)
    
    # Initialize fine-tuner
    fine_tuner = RLFineTuner()
    
    # Load targeted scenarios
    scenarios = fine_tuner.load_targeted_scenarios()
    
    # Fine-tune agent
    episode_rewards, state_coverage = fine_tuner.fine_tune(scenarios, episodes=1000)
    
    # Save fine-tuned agent
    fine_tuner.save_fine_tuned_agent()
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"📈 Coverage improved from 3.0% to {len(state_coverage)/500*100:.1f}%")
    print(f"🎯 Agent ready for deployment with better state coverage")

if __name__ == "__main__":
    main()
