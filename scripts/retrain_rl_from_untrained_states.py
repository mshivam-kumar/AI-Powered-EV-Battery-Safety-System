#!/usr/bin/env python3
"""
RL Agent Retraining Script - Using Untrained States
===================================================

This script uses the collected untrained states from the dashboard to create
targeted training scenarios for improving RL agent coverage.

Usage:
    python retrain_rl_from_untrained_states.py

Files:
    Input:  rl_untrained_states.json (from dashboard)
    Output: rl_retrained_agent.pkl (improved RL agent)
"""

import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from collections import defaultdict

class RLRetrainer:
    def __init__(self):
        self.ACTIONS = ['fast_charge', 'slow_charge', 'maintain', 'discharge', 'pause']
        self.ACTION_INDICES = {action: i for i, action in enumerate(self.ACTIONS)}
        
        # RL hyperparameters
        self.ALPHA = 0.1      # Learning rate
        self.GAMMA = 0.95     # Discount factor
        self.EPSILON = 0.2    # Exploration rate (lower for fine-tuning)
        
        # Load existing Q-table
        self.load_existing_q_table()
        
    def load_existing_q_table(self):
        """Load the existing RL agent Q-table"""
        models_dir = Path("../models")
        q_table_files = [
            "rl_robust_enhanced_v2_q_table.pkl",
            "rl_safety_focused_q_table.pkl",
            "rl_conservative_q_table.pkl"
        ]
        
        for q_file in q_table_files:
            q_path = models_dir / q_file
            if q_path.exists():
                try:
                    with open(q_path, 'rb') as f:
                        self.q_table = pickle.load(f)
                    print(f"✅ Loaded existing Q-table from {q_file}")
                    print(f"   Q-table shape: {self.q_table.shape}")
                    return
                except Exception as e:
                    print(f"❌ Failed to load {q_file}: {e}")
                    continue
        
        print("❌ No existing Q-table found. Please train base RL agent first.")
        exit(1)
    
    def load_untrained_states(self):
        """Load untrained states from dashboard logs"""
        untrained_file = Path("../rl_untrained_states.json")
        
        if not untrained_file.exists():
            print("❌ No untrained states file found. Run dashboard first to collect untrained states.")
            return []
        
        try:
            with open(untrained_file, 'r') as f:
                untrained_states = json.load(f)
            
            print(f"✅ Loaded {len(untrained_states)} untrained states")
            return untrained_states
        except Exception as e:
            print(f"❌ Failed to load untrained states: {e}")
            return []
    
    def analyze_untrained_patterns(self, untrained_states):
        """Analyze patterns in untrained states"""
        print("\n📊 UNTRAINED STATE ANALYSIS")
        print("=" * 50)
        
        # Group by safety priority
        high_priority = [s for s in untrained_states if s.get('safety_priority') == 'high']
        normal_priority = [s for s in untrained_states if s.get('safety_priority') == 'normal']
        
        print(f"🔴 High Priority States: {len(high_priority)}")
        print(f"🟡 Normal Priority States: {len(normal_priority)}")
        
        # Analyze state dimensions
        state_dims = defaultdict(set)
        for state in untrained_states:
            bins = state['state_bins']
            for i, bin_val in enumerate(bins):
                state_dims[i].add(bin_val)
        
        print(f"\n📈 State Space Coverage:")
        dim_names = ['c_rate', 'power', 'temp', 'soc', 'voltage', 'anomaly']
        for i, name in enumerate(dim_names):
            print(f"   {name}: {sorted(state_dims[i])}")
        
        return high_priority, normal_priority
    
    def compute_reward(self, soc_std, temp_std, voltage_std, is_anomaly, action_idx):
        """Enhanced reward function for untrained state scenarios"""
        action = self.ACTIONS[action_idx]
        
        # Start with neutral reward
        reward = 0.0
        
        # SAFETY FIRST - Massive penalties for dangerous actions
        if temp_std > 2.0 and action == 'fast_charge':
            return -10000.0  # Never fast charge when very hot
        
        if soc_std > 0.0 and action in ['fast_charge', 'slow_charge']:
            return -5000.0   # Never charge when battery is full
        
        if is_anomaly and action in ['fast_charge', 'slow_charge']:
            return -8000.0   # Never charge during anomalies
        
        # HIGH TEMPERATURE MANAGEMENT
        if temp_std > 1.5:  # High temperature
            if action == 'pause':
                reward = 500.0
            elif action == 'discharge':
                reward = 300.0
            elif action == 'maintain':
                reward = 100.0
            else:
                reward = -2000.0
            return reward
        
        # CRITICAL SOC MANAGEMENT
        if soc_std < -10.0:  # Very low SoC
            if action == 'slow_charge' and temp_std < 1.0:
                reward = 300.0
            elif action == 'fast_charge' and temp_std < 0.0:
                reward = 200.0
            else:
                reward = -1000.0
            return reward
        
        if soc_std > 0.0:  # High SoC
            if action == 'discharge':
                reward = 400.0
            elif action == 'maintain':
                reward = 200.0
            elif action == 'pause':
                reward = 100.0
            return reward
        
        # NORMAL CONDITIONS - Optimize for efficiency and safety
        if temp_std <= 0.0:  # Cool conditions
            if action == 'fast_charge' and soc_std < -2.0:
                reward = 100.0
            elif action == 'slow_charge':
                reward = 80.0
        elif temp_std <= 1.0:  # Normal temperature
            if action == 'slow_charge':
                reward = 50.0
            elif action == 'maintain':
                reward = 40.0
        
        return reward
    
    def train_on_untrained_states(self, untrained_states, episodes=1000):
        """Train RL agent specifically on untrained states"""
        print(f"\n🚀 TRAINING ON UNTRAINED STATES")
        print("=" * 50)
        
        trained_states = set()
        
        for episode in range(episodes):
            # Select an untrained state (prioritize high-priority ones)
            high_priority = [s for s in untrained_states if s.get('safety_priority') == 'high']
            
            if high_priority and np.random.random() < 0.7:  # 70% focus on high priority
                state_data = np.random.choice(high_priority)
            else:
                state_data = np.random.choice(untrained_states)
            
            # Extract state information
            state_bins = tuple(state_data['state_bins'])
            std_telemetry = state_data['standardized_telemetry']
            
            # Get current Q-values for this state
            if len(state_bins) == 6:  # Ensure correct dimensions
                try:
                    current_q = self.q_table[state_bins]
                    
                    # Choose action (epsilon-greedy)
                    if np.random.random() < self.EPSILON:
                        action_idx = np.random.randint(0, len(self.ACTIONS))
                    else:
                        action_idx = np.argmax(current_q)
                    
                    # Compute reward for this state-action pair
                    reward = self.compute_reward(
                        std_telemetry['soc'],
                        std_telemetry['temperature'], 
                        std_telemetry['voltage'],
                        std_telemetry['is_anomaly'],
                        action_idx
                    )
                    
                    # Q-learning update
                    # For untrained states, we assume next state is similar (simplified)
                    max_next_q = np.max(current_q)  # Simplified next state
                    
                    # Update Q-value
                    old_q = current_q[action_idx]
                    new_q = old_q + self.ALPHA * (reward + self.GAMMA * max_next_q - old_q)
                    self.q_table[state_bins][action_idx] = new_q
                    
                    trained_states.add(state_bins)
                    
                except IndexError:
                    # Skip states that don't fit Q-table dimensions
                    continue
            
            # Progress reporting
            if episode % 200 == 0:
                print(f"Episode {episode:4d} | Trained states: {len(trained_states):3d} | "
                      f"Last reward: {reward:8.1f}")
        
        print(f"\n✅ Training complete!")
        print(f"   Episodes: {episodes}")
        print(f"   Unique states trained: {len(trained_states)}")
        print(f"   Coverage: {len(trained_states)}/{len(untrained_states)} "
              f"({len(trained_states)/len(untrained_states)*100:.1f}%)")
        
        return len(trained_states)
    
    def save_retrained_agent(self):
        """Save the retrained RL agent"""
        output_file = Path("../models/rl_retrained_agent.pkl")
        
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(self.q_table, f)
            
            print(f"\n💾 Retrained RL agent saved to: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to save retrained agent: {e}")
            return False
    
    def validate_improvements(self, untrained_states):
        """Validate that previously untrained states now have non-zero Q-values"""
        print(f"\n🔍 VALIDATION - CHECKING IMPROVEMENTS")
        print("=" * 50)
        
        improved_count = 0
        total_states = len(untrained_states)
        
        for state_data in untrained_states:
            state_bins = tuple(state_data['state_bins'])
            
            if len(state_bins) == 6:
                try:
                    q_values = self.q_table[state_bins]
                    if not np.allclose(q_values, 0.0):
                        improved_count += 1
                except IndexError:
                    continue
        
        improvement_rate = (improved_count / total_states * 100) if total_states > 0 else 0
        
        print(f"✅ Improved states: {improved_count}/{total_states} ({improvement_rate:.1f}%)")
        
        if improvement_rate > 80:
            print("🎉 EXCELLENT: >80% of untrained states now have learned Q-values!")
        elif improvement_rate > 50:
            print("👍 GOOD: >50% of untrained states improved")
        else:
            print("⚠️  NEEDS MORE TRAINING: <50% improvement rate")
        
        return improvement_rate

def main():
    """Main retraining workflow"""
    print("🤖 RL AGENT RETRAINING FROM UNTRAINED STATES")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize retrainer
    retrainer = RLRetrainer()
    
    # Load untrained states
    untrained_states = retrainer.load_untrained_states()
    if not untrained_states:
        return
    
    # Analyze patterns
    high_priority, normal_priority = retrainer.analyze_untrained_patterns(untrained_states)
    
    # Train on untrained states
    trained_count = retrainer.train_on_untrained_states(untrained_states, episodes=2000)
    
    # Validate improvements
    improvement_rate = retrainer.validate_improvements(untrained_states)
    
    # Save retrained agent
    if retrainer.save_retrained_agent():
        print(f"\n🎯 RETRAINING SUMMARY")
        print("=" * 30)
        print(f"📊 Untrained states processed: {len(untrained_states)}")
        print(f"🎯 States successfully trained: {trained_count}")
        print(f"📈 Improvement rate: {improvement_rate:.1f}%")
        print(f"💾 New model saved: rl_retrained_agent.pkl")
        print(f"\n💡 NEXT STEPS:")
        print("   1. Update dashboard to use rl_retrained_agent.pkl")
        print("   2. Delete rl_untrained_states.json after validation")
        print("   3. Test dashboard with improved RL agent")
        print("   4. Monitor for new untrained states")

if __name__ == "__main__":
    main()
