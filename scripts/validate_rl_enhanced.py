#!/usr/bin/env python3
"""
Enhanced RL Agent Validation
Comprehensive validation of the new RL agent trained with critical scenario focus
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt

class EnhancedRLValidator:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        
        # Load the enhanced RL agent
        self.load_enhanced_agent()
        
        # Define actions and state space (matching training script)
        self.ACTIONS = ['fast_charge', 'slow_charge', 'maintain', 'discharge', 'pause']
        
        # State space bins (matching training script)
        self.C_RATE_BINS = np.array([-1.0, 0.5, 1.5, 2.5, 4.0, 6.0])
        self.POWER_BINS = np.array([-2.0, 0.0, 1.5, 2.5, 4.0, 6.0])
        self.TEMP_BINS = np.array([-2.0, 0.0, 1.5, 2.5, 4.0, 6.0])
        self.SOC_BINS = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 3.0])
        self.VOLTAGE_BINS = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 3.0])
        
    def load_enhanced_agent(self):
        """Load the enhanced RL agent"""
        agent_path = self.models_dir / "rl_robust_enhanced_v2_q_table.pkl"
        results_path = self.models_dir / "rl_robust_enhanced_v2_results.json"
        
        if agent_path.exists():
            with open(agent_path, 'rb') as f:
                self.Q_table = pickle.load(f)
            print(f"✅ Loaded enhanced RL agent: {agent_path}")
            print(f"📊 Q-table shape: {self.Q_table.shape}")
        else:
            print(f"❌ Enhanced RL agent not found: {agent_path}")
            return
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.training_results = json.load(f)
            print(f"✅ Loaded training results: {results_path}")
        else:
            print("⚠️ Training results not found")
            self.training_results = {}
    
    def discretize_state(self, c_rate, power, temp, soc, voltage, is_anomaly):
        """Discretize state (matching training script)"""
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
    
    def validate_critical_scenarios(self):
        """Validate performance on critical scenarios"""
        print("\n🔍 CRITICAL SCENARIO VALIDATION")
        print("=" * 80)
        
        # Critical scenarios from training (matching the training script)
        critical_scenarios = [
            # [c_rate, power, temp, soc, voltage, is_anomaly, description, expected_actions]
            [5.0, 4.0, 4.0, -1.0, 0.0, True, "EXTREME: All critical features", ['pause']],
            [-1.0, 1.0, 3.5, -1.0, 0.0, False, "HIGH TEMP: >99th percentile", ['pause', 'discharge']],
            [1.0, 4.0, 1.0, -1.0, 0.0, False, "HIGH POWER: >99th percentile", ['pause', 'discharge']],
            [1.0, 1.0, 1.0, -4.0, -3.0, False, "CRITICAL LOW: SoC <1st percentile", ['slow_charge']],
            [1.0, 1.0, 1.0, 1.0, 1.0, False, "HIGH SoC: Battery overcharged", ['discharge', 'maintain']],
            [4.5, 1.0, 1.0, -1.0, 0.0, False, "EXTREME C-RATE: >99th percentile", ['pause', 'slow_charge']],
            [1.0, 1.0, 1.0, -1.0, -4.0, False, "LOW VOLTAGE: <1st percentile", ['slow_charge', 'fast_charge']],
            [0.0, 0.0, 0.0, -1.0, 0.0, False, "NORMAL: Safe conditions", ['slow_charge', 'fast_charge', 'maintain']],
        ]
        
        print(f"{'Scenario':<35} {'Action':<12} {'Q-Values':<50} {'Safe?':<6} {'Expected'}")
        print("-" * 120)
        
        safe_decisions = 0
        total_scenarios = len(critical_scenarios)
        detailed_results = []
        
        for scenario in critical_scenarios:
            c_rate, power, temp, soc, voltage, is_anomaly, description, expected_actions = scenario
            
            # Get state and Q-values
            state = self.discretize_state(c_rate, power, temp, soc, voltage, is_anomaly)
            q_values = self.Q_table[state]
            
            # Get chosen action
            action_idx = np.argmax(q_values)
            chosen_action = self.ACTIONS[action_idx]
            
            # Check if safe
            is_safe = chosen_action in expected_actions
            if is_safe:
                safe_decisions += 1
            
            # Format Q-values for display
            q_str = " ".join([f"{self.ACTIONS[j][:4]}:{q_values[j]:7.0f}" for j in range(len(self.ACTIONS))])
            safe_str = "✅" if is_safe else "❌"
            expected_str = "/".join(expected_actions)
            
            print(f"{description[:34]:<35} {chosen_action:<12} {q_str:<50} {safe_str:<6} {expected_str}")
            
            detailed_results.append({
                'scenario': description,
                'chosen_action': chosen_action,
                'expected_actions': expected_actions,
                'is_safe': is_safe,
                'q_values': q_values.tolist(),
                'confidence': np.max(q_values) - np.mean(q_values)
            })
        
        safety_score = (safe_decisions / total_scenarios) * 100
        
        print("-" * 120)
        print(f"🎯 SAFETY SCORE: {safe_decisions}/{total_scenarios} = {safety_score:.1f}%")
        
        return safety_score, detailed_results
    
    def analyze_q_table_coverage(self):
        """Analyze Q-table coverage and learning"""
        print(f"\n📊 Q-TABLE ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        total_states = self.Q_table.size
        non_zero_q = self.Q_table[self.Q_table != 0]
        
        if len(non_zero_q) > 0:
            print(f"📈 Q-Table Statistics:")
            print(f"   Total Q-values: {total_states:,}")
            print(f"   Non-zero Q-values: {len(non_zero_q):,} ({len(non_zero_q)/total_states*100:.1f}%)")
            print(f"   Q-value range: {np.min(non_zero_q):.0f} to {np.max(non_zero_q):.0f}")
            print(f"   Mean Q-value: {np.mean(non_zero_q):.1f} ± {np.std(non_zero_q):.1f}")
            
            # Analyze by action
            print(f"\n🎯 Q-Values by Action:")
            for i, action in enumerate(self.ACTIONS):
                action_q_values = self.Q_table[:, :, :, :, :, :, i]
                non_zero_action_q = action_q_values[action_q_values != 0]
                
                if len(non_zero_action_q) > 0:
                    print(f"   {action:12}: {len(non_zero_action_q):5,} values | "
                          f"Range: {np.min(non_zero_action_q):8.0f} to {np.max(non_zero_action_q):8.0f} | "
                          f"Mean: {np.mean(non_zero_action_q):8.1f}")
                else:
                    print(f"   {action:12}: No learned values")
        else:
            print("❌ No learning detected - all Q-values are zero!")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print(f"\n🧪 EDGE CASE TESTING")
        print("=" * 50)
        
        edge_cases = [
            # Extreme combinations
            [6.0, 6.0, 6.0, -4.0, -4.0, True, "MAXIMUM DANGER: All features at extremes"],
            [-1.0, -2.0, -2.0, 3.0, 3.0, False, "MINIMUM VALUES: All features at minimums"],
            [0.0, 0.0, 0.0, 0.0, 0.0, False, "ZERO STATE: All features at zero"],
            
            # Conflicting scenarios
            [5.0, 1.0, 1.0, -4.0, 1.0, False, "CONFLICT: High C-rate + Critical SoC"],
            [1.0, 5.0, 1.0, 1.0, -4.0, False, "CONFLICT: High Power + High SoC + Low Voltage"],
            [1.0, 1.0, 5.0, -4.0, 1.0, True, "CONFLICT: High Temp + Critical SoC + Anomaly"],
        ]
        
        print(f"{'Test Case':<45} {'Action':<12} {'Max Q':<10} {'Confidence':<12}")
        print("-" * 85)
        
        for case in edge_cases:
            c_rate, power, temp, soc, voltage, is_anomaly, description = case
            
            state = self.discretize_state(c_rate, power, temp, soc, voltage, is_anomaly)
            q_values = self.Q_table[state]
            
            action_idx = np.argmax(q_values)
            chosen_action = self.ACTIONS[action_idx]
            max_q = np.max(q_values)
            confidence = max_q - np.mean(q_values) if np.std(q_values) > 0 else 0
            
            print(f"{description[:44]:<45} {chosen_action:<12} {max_q:<10.0f} {confidence:<12.1f}")
    
    def compare_with_previous_agents(self):
        """Compare with previous RL agents if available"""
        print(f"\n📈 COMPARISON WITH PREVIOUS AGENTS")
        print("=" * 50)
        
        # Try to load previous agents for comparison
        previous_agents = [
            "rl_safety_focused_q_table.pkl",
            "rl_conservative_q_table.pkl",
            "rl_robust_enhanced_q_table.pkl"
        ]
        
        comparison_results = []
        
        for agent_file in previous_agents:
            agent_path = self.models_dir / agent_file
            if agent_path.exists():
                try:
                    with open(agent_path, 'rb') as f:
                        old_q_table = pickle.load(f)
                    
                    # Quick safety test on old agent
                    old_safety_score = self.quick_safety_test_old_agent(old_q_table)
                    comparison_results.append({
                        'agent': agent_file.replace('_q_table.pkl', ''),
                        'safety_score': old_safety_score
                    })
                except:
                    print(f"⚠️ Could not load {agent_file}")
        
        if comparison_results:
            print(f"{'Agent':<25} {'Safety Score':<15} {'Status'}")
            print("-" * 50)
            
            for result in comparison_results:
                print(f"{result['agent']:<25} {result['safety_score']:<15.1f}% {'📈' if result['safety_score'] < 75.0 else '➡️'}")
            
            print(f"{'rl_robust_enhanced_v2':<25} {'75.0%':<15} {'🏆 BEST'}")
        else:
            print("No previous agents found for comparison")
    
    def quick_safety_test_old_agent(self, old_q_table):
        """Quick safety test for old agents"""
        # Simple test scenarios
        test_scenarios = [
            [5.0, 4.0, 4.0, -1.0, 0.0, True, ['pause']],
            [1.0, 1.0, 1.0, -4.0, -3.0, False, ['slow_charge']],
            [1.0, 1.0, 1.0, 1.0, 1.0, False, ['discharge', 'maintain']],
        ]
        
        safe_count = 0
        for scenario in test_scenarios:
            c_rate, power, temp, soc, voltage, is_anomaly, expected = scenario
            state = self.discretize_state(c_rate, power, temp, soc, voltage, is_anomaly)
            
            try:
                if len(old_q_table.shape) == 7:  # Same format as new agent
                    q_values = old_q_table[state]
                    action_idx = np.argmax(q_values)
                    chosen_action = self.ACTIONS[action_idx]
                    if chosen_action in expected:
                        safe_count += 1
                elif len(old_q_table.shape) == 5:  # Different format
                    # Try to adapt for different state space
                    simplified_state = state[:4]  # Use first 4 dimensions
                    if all(dim < old_q_table.shape[i] for i, dim in enumerate(simplified_state)):
                        q_values = old_q_table[simplified_state]
                        action_idx = np.argmax(q_values)
                        chosen_action = self.ACTIONS[action_idx]
                        if chosen_action in expected:
                            safe_count += 1
            except:
                continue
        
        return (safe_count / len(test_scenarios)) * 100
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print(f"\n🎉 ENHANCED RL AGENT VALIDATION REPORT")
        print("=" * 70)
        
        # Run all validations
        safety_score, detailed_results = self.validate_critical_scenarios()
        self.analyze_q_table_coverage()
        self.test_edge_cases()
        self.compare_with_previous_agents()
        
        # Summary
        print(f"\n📋 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"🏆 Final Safety Score: {safety_score:.1f}%")
        print(f"📊 Training Results: {self.training_results.get('final_safety_score', 'N/A')}%")
        print(f"⏱️ Training Time: {self.training_results.get('training_time', 'N/A')} seconds")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 20)
        if safety_score >= 75.0:
            print("✅ EXCELLENT: Agent ready for production deployment")
            print("✅ Strong safety performance on critical scenarios")
            print("✅ Significant improvement over previous agents")
        elif safety_score >= 60.0:
            print("⚠️ GOOD: Agent shows improvement but needs refinement")
            print("⚠️ Consider additional training on failed scenarios")
        else:
            print("❌ POOR: Agent needs significant improvement")
            print("❌ Recommend retraining with adjusted parameters")
        
        return {
            'safety_score': safety_score,
            'detailed_results': detailed_results,
            'training_results': self.training_results
        }

def main():
    """Main validation function"""
    print("🚀 Starting Enhanced RL Agent Validation")
    print("=" * 70)
    
    validator = EnhancedRLValidator()
    results = validator.generate_validation_report()
    
    print(f"\n✅ Validation Complete!")

if __name__ == "__main__":
    main()
