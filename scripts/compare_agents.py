#!/usr/bin/env python3
"""
Compare original and fine-tuned RL agents
"""

import json
import numpy as np
from pathlib import Path

def analyze_agent_coverage(agent_path: str) -> dict:
    """Analyze agent coverage"""
    try:
        with open(agent_path, 'r') as f:
            agent_data = json.load(f)
        
        q_table = agent_data['q_table']
        
        # Count visited states
        visited_states = 0
        total_q_updates = 0
        
        for soc in range(5):
            for temp in range(5):
                for ambient in range(5):
                    for voltage in range(4):
                        q_values = q_table[soc][temp][ambient][voltage]
                        if any(q != 0 for q in q_values):
                            visited_states += 1
                            total_q_updates += sum(1 for q in q_values if q != 0)
        
        coverage = visited_states / 500 * 100
        
        return {
            'visited_states': visited_states,
            'coverage': coverage,
            'total_q_updates': total_q_updates,
            'avg_q_updates_per_state': total_q_updates / visited_states if visited_states > 0 else 0
        }
        
    except FileNotFoundError:
        return None

def main():
    """Compare agents"""
    print("🔍 RL Agent Comparison Analysis")
    print("=" * 50)
    
    # Analyze original agent
    original_stats = analyze_agent_coverage('models/climate_aware_rl_agent.json')
    if original_stats:
        print(f"📊 Original Agent (climate_aware_rl_agent.json):")
        print(f"   • Visited States: {original_stats['visited_states']}")
        print(f"   • Coverage: {original_stats['coverage']:.1f}%")
        print(f"   • Total Q-updates: {original_stats['total_q_updates']}")
        print(f"   • Avg Q-updates/state: {original_stats['avg_q_updates_per_state']:.1f}")
    else:
        print("❌ Original agent not found")
    
    print()
    
    # Analyze fine-tuned agent
    fine_tuned_stats = analyze_agent_coverage('models/fine_tuned_rl_agent.json')
    if fine_tuned_stats:
        print(f"📊 Fine-tuned Agent (fine_tuned_rl_agent.json):")
        print(f"   • Visited States: {fine_tuned_stats['visited_states']}")
        print(f"   • Coverage: {fine_tuned_stats['coverage']:.1f}%")
        print(f"   • Total Q-updates: {fine_tuned_stats['total_q_updates']}")
        print(f"   • Avg Q-updates/state: {fine_tuned_stats['avg_q_updates_per_state']:.1f}")
    else:
        print("❌ Fine-tuned agent not found")
    
    print()
    
    # Compare improvements
    if original_stats and fine_tuned_stats:
        improvement = fine_tuned_stats['visited_states'] - original_stats['visited_states']
        coverage_improvement = fine_tuned_stats['coverage'] - original_stats['coverage']
        
        print(f"📈 IMPROVEMENTS:")
        print(f"   • Additional States: +{improvement}")
        print(f"   • Coverage Improvement: +{coverage_improvement:.1f}%")
        print(f"   • Total Q-updates: {fine_tuned_stats['total_q_updates'] - original_stats['total_q_updates']:+d}")
        
        if improvement > 0:
            print(f"✅ Fine-tuning successful! {improvement} new states learned")
        else:
            print(f"⚠️  Fine-tuning didn't add new states")
    
    print()
    print("🎯 RECOMMENDATION:")
    if fine_tuned_stats and fine_tuned_stats['coverage'] > original_stats['coverage']:
        print("   Use fine-tuned agent for better state coverage")
    else:
        print("   Original agent may be sufficient")

if __name__ == "__main__":
    main()
