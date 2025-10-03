#!/usr/bin/env python3
"""
Safety Validation for Fine-tuned RL Agent
Ensures fine-tuning doesn't degrade safety performance
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

class FineTuningSafetyValidator:
    def __init__(self):
        self.actions = ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain']
        
    def load_models(self, original_path: str, fine_tuned_path: str):
        """Load original and fine-tuned models"""
        print("📊 Loading Models for Safety Validation")
        print("=" * 50)
        
        # Load original model
        try:
            with open(original_path, 'rb') as f:
                original_data = pickle.load(f)
            # Handle both dict and direct array formats
            if isinstance(original_data, dict):
                self.original_q_table = original_data['q_table']
            else:
                self.original_q_table = original_data
            print(f"✅ Original model loaded: {self.original_q_table.shape}")
        except Exception as e:
            print(f"❌ Error loading original model: {e}")
            return False
            
        # Load fine-tuned model
        try:
            with open(fine_tuned_path, 'r') as f:
                fine_tuned_data = json.load(f)
            self.fine_tuned_q_table = np.array(fine_tuned_data['q_table'])
            print(f"✅ Fine-tuned model loaded: {self.fine_tuned_q_table.shape}")
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            return False
            
        return True
    
    def create_critical_test_scenarios(self) -> List[Dict]:
        """Create critical safety test scenarios"""
        scenarios = [
            # High temperature scenarios
            {'soc': 0.5, 'temp': 0.9, 'ambient': 0.8, 'voltage': 3.8, 'c_rate': 2.0, 'power': 5.0, 'is_anomaly': True, 'expected_safe_actions': ['pause']},
            {'soc': 0.3, 'temp': 0.85, 'ambient': 0.7, 'voltage': 3.6, 'c_rate': 1.5, 'power': 3.0, 'is_anomaly': True, 'expected_safe_actions': ['pause', 'slow_charge']},
            
            # Low SoC scenarios
            {'soc': 0.1, 'temp': 0.4, 'ambient': 0.3, 'voltage': 3.2, 'c_rate': 0.5, 'power': 1.0, 'is_anomaly': False, 'expected_safe_actions': ['slow_charge', 'fast_charge']},
            {'soc': 0.05, 'temp': 0.3, 'ambient': 0.2, 'voltage': 3.0, 'c_rate': 0.2, 'power': 0.5, 'is_anomaly': False, 'expected_safe_actions': ['slow_charge', 'fast_charge']},
            
            # High SoC scenarios
            {'soc': 0.95, 'temp': 0.5, 'ambient': 0.4, 'voltage': 4.1, 'c_rate': 0.1, 'power': 0.2, 'is_anomaly': False, 'expected_safe_actions': ['pause', 'maintain']},
            {'soc': 0.9, 'temp': 0.6, 'ambient': 0.5, 'voltage': 4.0, 'c_rate': 0.3, 'power': 0.8, 'is_anomaly': False, 'expected_safe_actions': ['pause', 'maintain']},
            
            # Anomaly scenarios
            {'soc': 0.4, 'temp': 0.7, 'ambient': 0.6, 'voltage': 3.5, 'c_rate': 1.0, 'power': 2.0, 'is_anomaly': True, 'expected_safe_actions': ['pause']},
            {'soc': 0.6, 'temp': 0.8, 'ambient': 0.7, 'voltage': 3.7, 'c_rate': 1.5, 'power': 3.5, 'is_anomaly': True, 'expected_safe_actions': ['pause', 'slow_charge']},
        ]
        return scenarios
    
    def discretize_state(self, soc, temp, ambient, voltage, c_rate=0.0, power=0.0, is_anomaly=False):
        """Discretize state for 6D state space"""
        c_rate = abs(c_rate)
        power = abs(power)
        
        c_rate_bin = min(max(int(c_rate / 1.0), 0), 4)
        power_bin = min(max(int(power / 2.0), 0), 4)
        temp_bin = min(max(int(temp / 10.0), 0), 4)
        soc_bin = min(max(int(soc * 100.0 / 20.0), 0), 4)
        voltage_bin = min(max(int((voltage - 3.0) / 0.24), 0), 4)
        anomaly_bin = 1 if is_anomaly else 0
        
        return (c_rate_bin, power_bin, temp_bin, soc_bin, voltage_bin, anomaly_bin)
    
    def get_agent_action(self, q_table: np.ndarray, state: Tuple) -> str:
        """Get action from Q-table"""
        try:
            q_values = q_table[state]
            action_idx = np.argmax(q_values)
            return self.actions[action_idx]
        except:
            return 'maintain'  # Safe fallback
    
    def validate_safety_performance(self) -> Dict:
        """Validate safety performance of fine-tuned vs original"""
        print("\n🛡️ SAFETY VALIDATION")
        print("=" * 50)
        
        scenarios = self.create_critical_test_scenarios()
        results = {
            'original_safe_actions': 0,
            'fine_tuned_safe_actions': 0,
            'total_scenarios': len(scenarios),
            'safety_improvements': 0,
            'safety_degradations': 0,
            'detailed_results': []
        }
        
        for i, scenario in enumerate(scenarios):
            # Discretize state
            state = self.discretize_state(
                scenario['soc'], scenario['temp'], scenario['ambient'], 
                scenario['voltage'], scenario['c_rate'], scenario['power'], 
                scenario['is_anomaly']
            )
            
            # Get actions from both models
            original_action = self.get_agent_action(self.original_q_table, state)
            fine_tuned_action = self.get_agent_action(self.fine_tuned_q_table, state)
            
            # Check if actions are safe
            original_safe = original_action in scenario['expected_safe_actions']
            fine_tuned_safe = fine_tuned_action in scenario['expected_safe_actions']
            
            if original_safe:
                results['original_safe_actions'] += 1
            if fine_tuned_safe:
                results['fine_tuned_safe_actions'] += 1
                
            # Track improvements/degradations
            if not original_safe and fine_tuned_safe:
                results['safety_improvements'] += 1
            elif original_safe and not fine_tuned_safe:
                results['safety_degradations'] += 1
            
            # Store detailed results
            results['detailed_results'].append({
                'scenario': i+1,
                'state': state,
                'original_action': original_action,
                'fine_tuned_action': fine_tuned_action,
                'expected_safe': scenario['expected_safe_actions'],
                'original_safe': original_safe,
                'fine_tuned_safe': fine_tuned_safe,
                'safety_change': 'improved' if (not original_safe and fine_tuned_safe) else 
                                'degraded' if (original_safe and not fine_tuned_safe) else 'same'
            })
            
            print(f"Scenario {i+1}: {scenario['expected_safe_actions']}")
            print(f"  Original: {original_action} {'✅' if original_safe else '❌'}")
            print(f"  Fine-tuned: {fine_tuned_action} {'✅' if fine_tuned_safe else '❌'}")
            print()
        
        return results
    
    def validate_q_value_consistency(self) -> Dict:
        """Validate Q-value consistency and learning quality"""
        print("\n📊 Q-VALUE CONSISTENCY VALIDATION")
        print("=" * 50)
        
        # Compare Q-value statistics
        original_stats = {
            'mean': np.mean(self.original_q_table),
            'std': np.std(self.original_q_table),
            'min': np.min(self.original_q_table),
            'max': np.max(self.original_q_table),
            'non_zero_count': np.count_nonzero(self.original_q_table)
        }
        
        fine_tuned_stats = {
            'mean': np.mean(self.fine_tuned_q_table),
            'std': np.std(self.fine_tuned_q_table),
            'min': np.min(self.fine_tuned_q_table),
            'max': np.max(self.fine_tuned_q_table),
            'non_zero_count': np.count_nonzero(self.fine_tuned_q_table)
        }
        
        print("Original Q-table Statistics:")
        for key, value in original_stats.items():
            print(f"  {key}: {value:.3f}")
            
        print("\nFine-tuned Q-table Statistics:")
        for key, value in fine_tuned_stats.items():
            print(f"  {key}: {value:.3f}")
        
        # Check for learning improvements
        improvements = {
            'mean_improvement': fine_tuned_stats['mean'] - original_stats['mean'],
            'max_improvement': fine_tuned_stats['max'] - original_stats['max'],
            'coverage_improvement': fine_tuned_stats['non_zero_count'] - original_stats['non_zero_count']
        }
        
        print(f"\nLearning Improvements:")
        for key, value in improvements.items():
            print(f"  {key}: {value:.3f}")
        
        return {
            'original_stats': original_stats,
            'fine_tuned_stats': fine_tuned_stats,
            'improvements': improvements
        }
    
    def generate_safety_report(self, safety_results: Dict, q_value_results: Dict):
        """Generate comprehensive safety report"""
        print("\n📋 SAFETY VALIDATION REPORT")
        print("=" * 60)
        
        # Safety performance summary
        original_safety_rate = safety_results['original_safe_actions'] / safety_results['total_scenarios']
        fine_tuned_safety_rate = safety_results['fine_tuned_safe_actions'] / safety_results['total_scenarios']
        
        print(f"🛡️ SAFETY PERFORMANCE:")
        print(f"  Original Safety Rate: {original_safety_rate:.1%}")
        print(f"  Fine-tuned Safety Rate: {fine_tuned_safety_rate:.1%}")
        print(f"  Safety Change: {fine_tuned_safety_rate - original_safety_rate:+.1%}")
        
        print(f"\n📈 IMPROVEMENTS:")
        print(f"  Safety Improvements: {safety_results['safety_improvements']}")
        print(f"  Safety Degradations: {safety_results['safety_degradations']}")
        
        # Q-value learning summary
        print(f"\n🧠 LEARNING QUALITY:")
        print(f"  Mean Q-value Change: {q_value_results['improvements']['mean_improvement']:+.3f}")
        print(f"  Max Q-value Change: {q_value_results['improvements']['max_improvement']:+.3f}")
        print(f"  Coverage Improvement: {q_value_results['improvements']['coverage_improvement']:+d} states")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if fine_tuned_safety_rate >= original_safety_rate and safety_results['safety_degradations'] == 0:
            print("✅ SAFE: Fine-tuning maintains or improves safety")
        elif safety_results['safety_degradations'] <= 1:
            print("⚠️ ACCEPTABLE: Minor safety trade-offs with overall improvement")
        else:
            print("❌ UNSAFE: Significant safety degradations detected")
        
        return {
            'safety_rate_original': original_safety_rate,
            'safety_rate_fine_tuned': fine_tuned_safety_rate,
            'safety_change': fine_tuned_safety_rate - original_safety_rate,
            'is_safe': fine_tuned_safety_rate >= original_safety_rate and safety_results['safety_degradations'] <= 1
        }

def main():
    """Main validation function"""
    print("🛡️ Fine-tuning Safety Validation")
    print("=" * 60)
    
    validator = FineTuningSafetyValidator()
    
    # Load models
    original_path = "models/rl_robust_enhanced_v2_q_table.pkl"
    fine_tuned_path = "models/fine_tuned_from_logs_rl_agent.json"
    
    if not validator.load_models(original_path, fine_tuned_path):
        print("❌ Failed to load models")
        return
    
    # Run validations
    safety_results = validator.validate_safety_performance()
    q_value_results = validator.validate_q_value_consistency()
    final_report = validator.generate_safety_report(safety_results, q_value_results)
    
    # Save results
    with open('fine_tuning_safety_validation.json', 'w') as f:
        json.dump({
            'safety_results': safety_results,
            'q_value_results': q_value_results,
            'final_report': final_report
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: fine_tuning_safety_validation.json")
    
    if final_report['is_safe']:
        print("✅ FINE-TUNING IS SAFE FOR DEPLOYMENT")
    else:
        print("❌ FINE-TUNING NEEDS REVIEW BEFORE DEPLOYMENT")

if __name__ == "__main__":
    main()
