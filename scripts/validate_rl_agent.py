#!/usr/bin/env python3
"""
RL Agent Validation - Check if agent is making smart battery management decisions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class RLValidator:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        
        # Same state/action space as training
        self.SOC_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        self.TEMP_BINS = [0.0, 20.0, 30.0, 40.0, 50.0, 100.0]
        self.AMBIENT_BINS = [0.0, 20.0, 30.0, 40.0, 60.0, 100.0]
        self.VOLTAGE_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
        self.ACTIONS = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
        
    def load_q_table(self, agent_name: str):
        """Load trained Q-table"""
        q_table_file = self.models_dir / f"{agent_name}_q_table.pkl"
        
        if not q_table_file.exists():
            print(f"❌ Q-table not found: {q_table_file}")
            return None
            
        with open(q_table_file, 'rb') as f:
            q_table = pickle.load(f)
        
        print(f"✅ Loaded Q-table: {q_table.shape}")
        return q_table
    
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
    
    def get_agent_action(self, q_table, soc, temp, ambient, voltage):
        """Get agent's recommended action for given state"""
        state = self.discretize_state(soc, temp, ambient, voltage)
        action_idx = np.argmax(q_table[state])
        return self.ACTIONS[action_idx], action_idx
    
    def test_critical_scenarios(self, q_table, agent_name: str):
        """Test agent behavior in critical battery scenarios"""
        print(f"\n🧪 Testing {agent_name} in Critical Scenarios:")
        print("-" * 60)
        
        # Define critical test scenarios (CORRECTED THRESHOLDS for standardized data)
        scenarios = [
            # [soc, temp, ambient, voltage, expected_behavior, description]
            [-5.0, 0.5, 0.0, 0.0, ['slow_charge', 'pause'], "Low SoC, Normal Temp"],
            [-5.0, 2.0, 1.0, 0.0, ['pause'], "Low SoC, HIGH TEMP - Should pause!"],
            [0.01, 0.5, 0.0, 1.0, ['maintain', 'discharge'], "High SoC, Normal Temp"],
            [0.01, 2.0, 1.0, 1.0, ['pause', 'discharge'], "High SoC, HIGH TEMP - Dangerous!"],
            [-3.0, 0.5, 0.0, 0.5, ['slow_charge', 'fast_charge'], "Optimal Range - Can charge"],
            [-15.0, 0.5, 0.0, 0.0, ['slow_charge'], "CRITICAL Low SoC - Must charge safely"],
            [0.01, 1.8, 1.0, 1.0, ['pause', 'discharge'], "CRITICAL High SoC + Temp"],
            [-8.0, -0.5, -0.5, 0.0, ['fast_charge', 'slow_charge'], "Cold battery - Can fast charge"],
        ]
        
        results = []
        
        for i, (soc, temp, ambient, voltage, expected, description) in enumerate(scenarios):
            action, action_idx = self.get_agent_action(q_table, soc, temp, ambient, voltage)
            
            # Check if action is appropriate
            is_safe = action in expected
            safety_status = "✅ SAFE" if is_safe else "❌ UNSAFE"
            
            print(f"Scenario {i+1}: {description}")
            print(f"   State: SoC={soc:.1f}, Temp={temp:.1f}, Ambient={ambient:.1f}, Voltage={voltage:.1f}")
            print(f"   Agent Action: {action} {safety_status}")
            print(f"   Expected: {expected}")
            print()
            
            results.append({
                'scenario': description,
                'soc': soc,
                'temp': temp,
                'ambient': ambient,
                'voltage': voltage,
                'agent_action': action,
                'expected_actions': expected,
                'is_safe': is_safe
            })
        
        # Calculate safety score
        safety_score = sum([r['is_safe'] for r in results]) / len(results)
        print(f"🎯 Safety Score: {safety_score:.1%} ({sum([r['is_safe'] for r in results])}/{len(results)} scenarios)")
        
        return results, safety_score
    
    def analyze_action_patterns(self, q_table, agent_name: str):
        """Analyze what actions the agent prefers in different conditions"""
        print(f"\n📊 Analyzing {agent_name} Action Patterns:")
        print("-" * 50)
        
        # Sample different state combinations
        action_counts = {action: 0 for action in self.ACTIONS}
        temp_action_map = {}
        soc_action_map = {}
        
        # Test across state space
        n_samples = 1000
        for _ in range(n_samples):
            soc = np.random.uniform(0, 1)
            temp = np.random.uniform(0, 1)
            ambient = np.random.uniform(0, 1)
            voltage = np.random.uniform(0, 1)
            
            action, _ = self.get_agent_action(q_table, soc, temp, ambient, voltage)
            action_counts[action] += 1
            
            # Track patterns
            temp_range = "High" if temp > 0.7 else "Medium" if temp > 0.3 else "Low"
            soc_range = "High" if soc > 0.7 else "Medium" if soc > 0.3 else "Low"
            
            if temp_range not in temp_action_map:
                temp_action_map[temp_range] = {action: 0 for action in self.ACTIONS}
            temp_action_map[temp_range][action] += 1
            
            if soc_range not in soc_action_map:
                soc_action_map[soc_range] = {action: 0 for action in self.ACTIONS}
            soc_action_map[soc_range][action] += 1
        
        # Print overall action distribution
        print("Overall Action Distribution:")
        for action, count in action_counts.items():
            percentage = count / n_samples * 100
            print(f"   {action:12}: {count:3d} ({percentage:5.1f}%)")
        
        # Print temperature-based patterns
        print("\nAction by Temperature:")
        for temp_range, actions in temp_action_map.items():
            total = sum(actions.values())
            print(f"   {temp_range} Temp:")
            for action, count in actions.items():
                if count > 0:
                    percentage = count / total * 100
                    print(f"      {action:12}: {percentage:5.1f}%")
        
        # Print SoC-based patterns
        print("\nAction by State of Charge:")
        for soc_range, actions in soc_action_map.items():
            total = sum(actions.values())
            print(f"   {soc_range} SoC:")
            for action, count in actions.items():
                if count > 0:
                    percentage = count / total * 100
                    print(f"      {action:12}: {percentage:5.1f}%")
        
        return action_counts, temp_action_map, soc_action_map
    
    def validate_learning_progress(self):
        """Compare different RL agents to see learning improvement"""
        print("🔍 RL Agent Validation & Learning Analysis")
        print("=" * 60)
        
        agents = ['rl_safety_focused', 'rl_conservative']  # Test new corrected agents
        
        all_results = {}
        
        for agent_name in agents:
            print(f"\n🤖 Validating {agent_name.upper()}...")
            
            # Load Q-table
            q_table = self.load_q_table(agent_name)
            if q_table is None:
                continue
            
            # Test critical scenarios
            scenario_results, safety_score = self.test_critical_scenarios(q_table, agent_name)
            
            # Analyze action patterns
            action_dist, temp_patterns, soc_patterns = self.analyze_action_patterns(q_table, agent_name)
            
            all_results[agent_name] = {
                'safety_score': safety_score,
                'scenario_results': scenario_results,
                'action_distribution': action_dist,
                'temp_patterns': temp_patterns,
                'soc_patterns': soc_patterns
            }
        
        # Compare agents
        if len(all_results) >= 2:
            print(f"\n🏆 Agent Comparison:")
            print("-" * 30)
            for agent_name, results in all_results.items():
                print(f"{agent_name:12}: Safety Score = {results['safety_score']:.1%}")
        
        # Save validation results
        results_file = self.models_dir / "rl_validation_results.json"
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for agent_name, results in all_results.items():
            json_results[agent_name] = {
                'safety_score': float(results['safety_score']),
                'scenario_results': results['scenario_results'],
                'action_distribution': {k: int(v) for k, v in results['action_distribution'].items()}
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Validation results saved: {results_file}")
        
        return all_results

def main():
    """Main validation function"""
    validator = RLValidator()
    results = validator.validate_learning_progress()
    
    # Summary
    print(f"\n✅ RL Agent Validation Complete!")
    
    if results:
        best_agent = max(results.keys(), key=lambda k: results[k]['safety_score'])
        best_score = results[best_agent]['safety_score']
        print(f"🏆 Best Agent: {best_agent} (Safety: {best_score:.1%})")
        
        # Check if agents are learning properly
        if best_score >= 0.75:
            print("🎯 Agent is making GOOD battery management decisions!")
        elif best_score >= 0.5:
            print("⚠️  Agent is making OKAY decisions, but needs improvement")
        else:
            print("❌ Agent is making POOR decisions - needs retraining")
    else:
        print("❌ No trained agents found for validation")

if __name__ == "__main__":
    main()

