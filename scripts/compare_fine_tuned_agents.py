#!/usr/bin/env python3
"""
Compare original vs fine-tuned RL agents
"""

import json
import numpy as np
from pathlib import Path

def load_agent(agent_path: str):
    """Load RL agent from file"""
    try:
        with open(agent_path, 'r') as f:
            agent_data = json.load(f)
        
        q_table = np.array(agent_data['q_table'])
        actions = agent_data.get('actions', ['fast_charge', 'slow_charge', 'pause', 'discharge', 'maintain'])
        agent_type = agent_data.get('agent_type', 'unknown')
        
        return q_table, actions, agent_type
    except Exception as e:
        print(f"❌ Failed to load {agent_path}: {e}")
        return None, None, None

def analyze_q_table(q_table, agent_name):
    """Analyze Q-table properties"""
    print(f"\n📊 {agent_name} Analysis:")
    print(f"   • Q-table shape: {q_table.shape}")
    print(f"   • Total states: {q_table.size}")
    
    # Count non-zero Q-values
    non_zero_count = np.count_nonzero(q_table)
    total_states = q_table.size
    coverage = (non_zero_count / total_states) * 100
    
    print(f"   • Non-zero Q-values: {non_zero_count:,}")
    print(f"   • Coverage: {coverage:.1f}%")
    
    # Q-value statistics
    q_values = q_table[q_table != 0]
    if len(q_values) > 0:
        print(f"   • Q-value range: {q_values.min():.3f} to {q_values.max():.3f}")
        print(f"   • Q-value mean: {q_values.mean():.3f}")
        print(f"   • Q-value std: {q_values.std():.3f}")
    
    # Action distribution
    action_counts = np.count_nonzero(q_table, axis=-1)
    print(f"   • States with learned actions: {np.count_nonzero(action_counts)}")
    print(f"   • Max actions per state: {action_counts.max()}")
    
    return {
        'coverage': coverage,
        'non_zero_count': non_zero_count,
        'total_states': total_states,
        'q_value_stats': {
            'min': q_values.min() if len(q_values) > 0 else 0,
            'max': q_values.max() if len(q_values) > 0 else 0,
            'mean': q_values.mean() if len(q_values) > 0 else 0,
            'std': q_values.std() if len(q_values) > 0 else 0
        }
    }

def compare_agents():
    """Compare different RL agents"""
    print("🔍 RL Agent Comparison")
    print("=" * 50)
    
    # Load agents
    agents = {
        'Original Climate-Aware': 'models/climate_aware_rl_agent.json',
        'Fine-tuned from Logs': 'models/fine_tuned_from_logs_rl_agent.json'
    }
    
    results = {}
    
    for name, path in agents.items():
        if Path(path).exists():
            q_table, actions, agent_type = load_agent(path)
            if q_table is not None:
                results[name] = analyze_q_table(q_table, name)
                print(f"   • Agent type: {agent_type}")
            else:
                print(f"❌ Failed to load {name}")
        else:
            print(f"❌ {name} not found at {path}")
    
    # Comparison summary
    if len(results) >= 2:
        print(f"\n📈 COMPARISON SUMMARY:")
        print(f"=" * 50)
        
        original = results.get('Original Climate-Aware', {})
        fine_tuned = results.get('Fine-tuned from Logs', {})
        
        if original and fine_tuned:
            coverage_improvement = fine_tuned['coverage'] - original['coverage']
            print(f"📊 Coverage Improvement: {coverage_improvement:+.1f}%")
            print(f"   • Original: {original['coverage']:.1f}%")
            print(f"   • Fine-tuned: {fine_tuned['coverage']:.1f}%")
            
            q_improvement = fine_tuned['q_value_stats']['mean'] - original['q_value_stats']['mean']
            print(f"📊 Q-value Improvement: {q_improvement:+.3f}")
            print(f"   • Original mean: {original['q_value_stats']['mean']:.3f}")
            print(f"   • Fine-tuned mean: {fine_tuned['q_value_stats']['mean']:.3f}")
            
            if coverage_improvement > 0:
                print(f"✅ Fine-tuning improved coverage by {coverage_improvement:.1f}%")
            else:
                print(f"⚠️ Fine-tuning did not improve coverage")
            
            if q_improvement > 0:
                print(f"✅ Fine-tuning improved Q-values by {q_improvement:.3f}")
            else:
                print(f"⚠️ Fine-tuning did not improve Q-values")

def main():
    """Main comparison function"""
    compare_agents()

if __name__ == "__main__":
    main()


